from sentence_transformers import SentenceTransformer, models, evaluation
from sentence_transformers.evaluation import SentenceEvaluator
from torch import nn
import torch
from datasets import DatasetDict
from transformers import AutoModel, AutoTokenizer
from absl import logging
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from itertools import combinations
from itertools import groupby
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import logging
import random
import csv
import itertools
import pickle as pckl
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import numpy as np

from eval.hiatus.attribution import metrics
from author_attribution.sbert_luar import LuarTransformer, LuarSentenceTransformer


def aggregate_key_values(list_of_pairs):
    list_of_pairs.sort(key=lambda x: x[0])  # Sort the data based on the key
    d = {}
    for key, group in groupby(list_of_pairs, key=lambda x: x[0]):
        d[key] = [item[1] for item in group]
    
    return d

def compute_pairwise_selection(model, query_df, candidates_df, query_authors=None, candidate_authors=None, batch_size=32):

    queries = [row['fullText'] for idx, row in query_df.iterrows()]
    candidates = [row['fullText'] for idx, row in candidates_df.iterrows()]
    
    if query_authors ==None:
        query_authors = query_df.authorID.unique()
    
    if candidate_authors ==None:
        candidate_authors = candidates_df.authorSetID.unique()

    q_embeddings = model.encode(
        queries,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    
    c_embeddings = model.encode(
        candidates,
        batch_size=batch_size,
        convert_to_numpy=True,
    )

    query_to_candidate_docs_cos_sim = cosine_similarity(q_embeddings, c_embeddings)
        
    q_author_to_doc_id = aggregate_key_values(list(zip(query_df.authorID.tolist(), range(len(query_df.documentID)))))
    c_author_to_doc_id = aggregate_key_values(list(zip(candidates_df.authorSetID.tolist(), range(len(candidates_df.documentID)))))
    
    #The final output matrix
    author_pairwise_sim = np.zeros(shape=(len(query_authors), len(candidate_authors)))
    for i in tqdm(range(len(query_authors))):
        for j in range(len(candidate_authors)):
            #get the corresponding candidate and query author's document indices
            query_author_docs_indices     = q_author_to_doc_id[query_authors[i]]
            candidate_author_docs_indices = c_author_to_doc_id[candidate_authors[j]]
            cos_sims = [query_to_candidate_docs_cos_sim[i,j] for i, j in itertools.product(query_author_docs_indices, candidate_author_docs_indices)]

            #One can use max, min, avg, etc..
            avg_sim = np.mean(cos_sims) #for now avg
            author_pairwise_sim[i][j] = avg_sim

    return author_pairwise_sim, query_authors, candidate_authors

class EER_Evaluator(SentenceEvaluator):

    def __init__(
        self,
        query_df,
        candidates_df,
        ground_truth_assignment,
        query_authors,
        candidate_authors,
        batch_size=32,
        as_predictor=False,
        with_filtering=False,
        mask=None

    ):
    
        self.query_df = query_df
        self.candidates_df = candidates_df
        self.ground_truth_assignment = ground_truth_assignment
        self.query_authors = query_authors
        self.candidate_authors = candidate_authors
        self.batch_size = batch_size


        self.with_filtering = with_filtering
        self.mask = mask
        self.as_predictor= as_predictor
        
        self.csv_file = "eer_results.csv"
        self.csv_headers = [
            "epoch",
            "steps",
            "eer",
            "auc"
        ]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:

        author_pairwise_sim, query_authors, candidate_authors = compute_pairwise_selection(model, self.query_df, self.candidates_df, self.query_authors, self.candidate_authors)

        if self.with_filtering: #apply the mask
            print('Apply filtering step...')
            for i, q_author in enumerate(query_authors):
                assigned_candidate_authors = self.mask[q_author]
                for j, c_author in enumerate(candidate_authors):
                    if c_author not in assigned_candidate_authors:
                        author_pairwise_sim[i,j] = 0


        #This code is taken from the official hiatus eval
        fpr, fnr, thresh_det = metrics.det(author_pairwise_sim, self.ground_truth_assignment)
        eer_metric = metrics.eer(fpr, fnr)
        auc_metric = metrics.auc(author_pairwise_sim, self.ground_truth_assignment)

        print('steps {}, eer: {}, auc: {}'.format(steps, round(eer_metric, 3), round(auc_metric, 3)))

        #Log the err for the corresponding epoch and step
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
    
                writer.writerow(
                    [
                        epoch,
                        steps,
                        eer_metric,
                        auc_metric
                    ]
                )

        #if the evaluator was run for final prediction and evaluation, save the output matrix
        if self.as_predictor:
            return eer_metric, auc_metric, author_pairwise_sim, query_authors, candidate_authors
        else:
            return 1-eer_metric # we return 1-error because the library save the model with the highest value

class PAUSIT_MODEL:

    def __init__(self, training_dir, valid_dir, output_dir, run_id, query_identifier, candidate_identifier, ratio, model_path=None):
        
        #----------------
        #The logic in Luar_Weg will load the query and candidate dataframe that we will consider as a development set
        #super().__init__(valid_dir + '/data', output_dir, run_id, query_identifier, candidate_identifier, ratio)
        self.input_dir  = training_dir
        self.output_dir = output_dir
        self.run_id = run_id

        queries_fname, candidates_fname = PAUSIT_MODEL.get_file_names(valid_dir + '/data')
        self.query_df = pd.read_json(queries_fname, lines=True)
        self.candidates_df = pd.read_json(candidates_fname, lines=True)
        self.query_df = PAUSIT_MODEL.modify_df(data=self.query_df, author_identifier='authorIDs', ratio=ratio)

        #--------------------

        self.prefix = PAUSIT_MODEL.get_prefix(input_dir=training_dir + '/data')


        #Load the training dataframes
        if training_dir != None:
            queries_fname, candidates_fname = PAUSIT_MODEL.get_file_names(training_dir + '/data/')
            self.training_query_df = pd.read_json(queries_fname, lines=True)
            self.training_candidates_df = pd.read_json(candidates_fname, lines=True)
            self.training_query_df = PAUSIT_MODEL.modify_df(data=self.training_query_df, author_identifier='authorIDs', ratio=ratio)

            self.training_candidates_df['authorSetID'] = self.training_candidates_df.authorSetIDs.apply(lambda x: x[0][2:-3] if x[0].startswith("(") else x[0])
            self.training_query_df['authorID'] = self.training_query_df.authorIDs.apply(lambda x:  x[0][2:-3] if x[0].startswith("(") else x[0])
            
            self.candidates_df['authorSetID'] = self.candidates_df.authorSetIDs.apply(lambda x: x[0][2:-3] if x[0].startswith("(") else x[0])
            self.query_df['authorID'] = self.query_df.authorIDs.apply(lambda x: x[0][2:-3] if x[0].startswith("(") else x[0])
            
            self.training_ground_truth_path = training_dir + '/groundtruth/'
            self.valid_ground_truth_path = valid_dir + '/groundtruth/'

            self.training_ground_truth_assignment = np.load(open(self.training_ground_truth_path + self.prefix + '_TA2_groundtruth.npy', 'rb'))
            self.training_candidate_authors = [a[2:-3] for a in  open(self.training_ground_truth_path + self.prefix + '_TA2_candidate-labels.txt').read().split('\n')][:-1]
            self.training_query_authors = [a[2:-3] for a in  open(self.training_ground_truth_path + self.prefix + '_TA2_query-labels.txt').read().split('\n')][:-1]


        #Load the model if there is a path provided. This is only when we want to predict
        self.model = self.load_model(model_path) if model_path else None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_model(self, model_path):
        return SentenceTransformer(model_path)
    
    
    def get_file_names(input_dir:str):
        queries_fname = glob(os.path.join(input_dir, "*queries*"))[0]
        candidates_fname = glob(os.path.join(input_dir, "*candidates*"))[0]
        return queries_fname, candidates_fname

    def get_prefix(input_dir):
        queries_fname, candidates_fname = PAUSIT_MODEL.get_file_names(input_dir)
        prefix = queries_fname.split(os.sep)[-1]
        return prefix[:prefix.find('_TA2')]

    def modify_df(data, author_identifier, ratio):

        raw_data = []
        for _, row in data.iterrows():
            document_id = row['documentID']
            author_id = row[author_identifier]
            text = row['fullText'].replace('<PERSON>','')
            start = 0
            modifier = len(text)//ratio
            while True:
                end = start + modifier
                if end > len(text):
                    break
                raw_data.append({'documentID': document_id, author_identifier: author_id, 'fullText': text[start:end]})
                start = end
            
        return pd.DataFrame.from_dict(raw_data)
    
    def save_output_files(prefix, output_dir, run_id, output_matrix, authorIDs, authorSetIDs):
        output_array_path = os.path.join(output_dir , f"{prefix}_TA2_query_candidate_attribution_scores_{run_id}.npy")
        output_query_labels_path = os.path.join(output_dir , f"{prefix}_TA2_query_candidate_attribution_query_labels_{run_id}.txt")
        output_candidate_labels_path = os.path.join(output_dir , f"{prefix}_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt")
        with open(output_array_path, 'wb') as f:
            np.save(f, output_matrix)
        
        with open(output_query_labels_path, 'w') as f:
            for line in authorIDs:
                f.write(f"('{line}',)\n")
        
        with open(output_candidate_labels_path, 'w') as f:
            for line in authorSetIDs:
                f.write(f"('{line}',)\n")

    def split_training(self, split_size=0.15):
        ground_truth_assignment = np.load(open(self.training_ground_truth_path + self.prefix + '_TA2_groundtruth.npy', 'rb'))
        candidate_authors = [a[2:-3] for a in  open(self.training_ground_truth_path + self.prefix + '_TA2_candidate-labels.txt').read().split('\n')][:-1]
        query_authors = [a[2:-3] for a in  open(self.training_ground_truth_path + self.prefix + '_TA2_query-labels.txt').read().split('\n')][:-1]


        q_training_authors_idx, q_valid_authors_idx = train_test_split(range(len(query_authors)), test_size=split_size)

        #c_training_authors_idx, c_valid_authors_idx = train_test_split(range(len(candidate_authors)), test_size=0.15)
        #print('#c training authors {}, #c valid_authors {}'.format(len(c_training_authors_idx), len(c_valid_authors_idx)))

        #Fill the candidate authors with the authors matching the query author
        c_training_authors_idx = []
        for q_i in q_training_authors_idx:
            c_training_authors_idx+= [j for j, a in enumerate(ground_truth_assignment[q_i]) if a==1]

        c_valid_authors_idx = []
        for q_i in q_valid_authors_idx:
            c_valid_authors_idx+= [j for j, a in enumerate(ground_truth_assignment[q_i]) if a==1]


        #split the rest of the candidate authors into training and valid
        rest_of_candidate_authors = [i for i, a in enumerate(candidate_authors) if i not in c_training_authors_idx and i not in c_valid_authors_idx]
        rest_c_training_authors, rest_c_valid_authors = train_test_split(rest_of_candidate_authors, test_size=split_size)

        c_training_authors_idx += rest_c_training_authors
        c_valid_authors_idx += rest_c_valid_authors

        #make sure there is no redundency
        print('#q training authors {}, #q valid_authors {}'.format(len(q_training_authors_idx), len(q_valid_authors_idx)))
        print('#c training authors {}, #c valid_authors {}'.format(len(c_training_authors_idx), len(c_valid_authors_idx)))

        c_training_authors_idx = list(set(c_training_authors_idx))
        c_valid_authors_idx = list(set(c_valid_authors_idx))

        print('#q training authors {}, #q valid_authors {}'.format(len(q_training_authors_idx), len(q_valid_authors_idx)))
        print('#c training authors {}, #c valid_authors {}'.format(len(c_training_authors_idx), len(c_valid_authors_idx)))

        training_gt_assignment = np.zeros(shape=(len(q_training_authors_idx), len(c_training_authors_idx)))
        valid_gt_assignment    = np.zeros(shape=(len(q_valid_authors_idx), len(c_valid_authors_idx)))
        
        for i in range(len(q_training_authors_idx)):
            for j in range(len(c_training_authors_idx)):
                ii = q_training_authors_idx[i]
                jj = c_training_authors_idx[j]
                training_gt_assignment[i,j] = ground_truth_assignment[ii,jj]

        for i in range(len(q_valid_authors_idx)):
            for j in range(len(c_valid_authors_idx)):
                ii = q_valid_authors_idx[i]
                jj = c_valid_authors_idx[j]
                valid_gt_assignment[i,j] = ground_truth_assignment[ii,jj]

        print('# training {}, valid {}'.format(training_gt_assignment.shape, valid_gt_assignment.shape))
        print('# training matches {}, valid matches {}'.format(np.sum(training_gt_assignment), np.sum(valid_gt_assignment)))
        return [query_authors[i] for i in q_training_authors_idx], [candidate_authors[i] for i in c_training_authors_idx], training_gt_assignment, [query_authors[i] for i in q_valid_authors_idx], [candidate_authors[i] for i in c_valid_authors_idx], valid_gt_assignment


    def build_luar_based_model(self):
        #style_model = MyTransformer("./data/models/LUAR-MUD/", tokenizer_args = {'trust_remote_code':True}, model_args = {'trust_remote_code':True})
        style_model = LuarTransformer("rrivera1849/LUAR-MUD", tokenizer_args = {'trust_remote_code':True}, model_args = {'trust_remote_code':True})
                        
        dense_model = models.Dense(
            in_features=512,
            out_features=128,
            activation_function=nn.Tanh(),
        )

        return LuarSentenceTransformer(modules=[style_model, dense_model])

    def build_model(self, model_size=128):
        style_embeddings = SentenceTransformer('AnnaWegmann/Style-Embedding')
        pooling_model = models.Pooling(768)
        
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=model_size,
            activation_function=nn.Tanh(),
        )

        return SentenceTransformer(modules=[style_embeddings, pooling_model, dense_model])

    def create_genre_training_pairs(self, query_df, candidates_df, query_authors, candidate_authors, ground_truth_assignment):
        pass

    def create_training_pairs(self, query_df, candidates_df, query_authors, candidate_authors, ground_truth_assignment):
        training_pairs = []

        # create pairs of queries and candidates that belong to the same author
        c_authors_docs_df = candidates_df.groupby('authorSetID').agg({'fullText': lambda x: list(x), 'documentID': lambda x: list(x)}).reset_index()
        q_authors_docs_df = query_df.groupby('authorID').agg({'fullText': lambda x: list(x), 'documentID': lambda x: list(x)}).reset_index()
        
        training_pairs+= [InputExample(texts=[p[0], p[1]]) for idx, row in q_authors_docs_df.iterrows() for p in list(combinations(row['fullText'], 2))]
        training_pairs+= [InputExample(texts=[p[0], p[1]]) for idx, row in c_authors_docs_df.iterrows() for p in list(combinations(row['fullText'], 2))]


        #create more training pairs from the candidates to the queries
        c_authors_docs = {row['authorSetID']: row['fullText'] for i, row in c_authors_docs_df.iterrows()}
        q_authors_docs = {row['authorID']: row['fullText'] for i,row in q_authors_docs_df.iterrows()}

                
        author_pairs = [(query_authors[p[0]], candidate_authors[p[1]]) for p in np.argwhere(ground_truth_assignment == 1)]
        c_doc_q_doc_pairs = []
        for q_author, c_author in author_pairs:
            c_author_docs = c_authors_docs[c_author]
            q_author_docs = q_authors_docs[q_author]
            for p in itertools.product(c_author_docs, q_author_docs): #TODO: problem of picking queries from the same author to be negatives
                c_doc_q_doc_pairs.append(InputExample(texts=[p[0], p[1]]))

        print("# pairs of q and c: ", len(c_doc_q_doc_pairs))
        print("# pairs of qs and cs: ", len(training_pairs))

        training_pairs += c_doc_q_doc_pairs

        return training_pairs

    def train(self, training_epochs=3, luar_based=False, evaluation_steps=100, valid_from_training=False, batch_size=32):

        if luar_based:
            self.model = self.build_luar_based_model()
        else:
            self.model = self.build_model()

        # Evalaute the model before training ===================
        ground_truth_assignment = np.load(open(self.valid_ground_truth_path + self.prefix + '_TA2_groundtruth.npy', 'rb'))
        candidate_authors = [a[2:-3] for a in  open(self.valid_ground_truth_path + self.prefix + '_TA2_candidate-labels.txt').read().split('\n')][:-1]
        query_authors = [a[2:-3] for a in  open(self.valid_ground_truth_path + self.prefix + '_TA2_query-labels.txt').read().split('\n')][:-1]

        eer_test_evaluator = EER_Evaluator(self.query_df, self.candidates_df, 
            ground_truth_assignment, query_authors, candidate_authors, batch_size=batch_size)

        print('Test EER before training', self.model.evaluate(eer_test_evaluator))
        #=========================================================


        if valid_from_training:
            #split the training into training and validation
            q_author_training, c_author_training, gt_assignment_training, q_author_valid, c_author_valid, gt_assignment_valid = self.split_training()
            
            valid_query_df = self.training_query_df[self.training_query_df.authorID.isin(q_author_valid)]
            valid_candidate_df =  self.training_candidates_df[self.training_candidates_df.authorSetID.isin(c_author_valid)]
            train_query_df = self.training_query_df[self.training_query_df.authorID.isin(q_author_training)]
            train_candidate_df =  self.training_candidates_df[self.training_candidates_df.authorSetID.isin(c_author_training)]

            training_pairs = self.create_training_pairs(train_query_df, train_candidate_df, 
            q_author_training, c_author_training, gt_assignment_training)
        
            #eer_train_evaluator = EER_Evaluator(train_query_df, train_candidate_df, gt_assignment_training, q_author_training, c_author_training, batch_size=batch_size)
            eer_valid_evaluator = EER_Evaluator(valid_query_df, valid_candidate_df, 
                gt_assignment_valid, q_author_valid, c_author_valid, batch_size=batch_size)

            print('Valid EER before training', self.model.evaluate(eer_valid_evaluator))


        else:
            training_pairs = self.create_training_pairs(self.training_query_df, self.training_candidates_df, 
            self.training_query_authors, self.training_candidate_authors, self.training_ground_truth_assignment)
            
            eer_valid_evaluator = eer_test_evaluator



        # Define your train dataset, the dataloader and the train loss
        train_dataloader = DataLoader(training_pairs, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        
        #save training data for reference
        pckl.dump(training_pairs, open(self.output_dir + '/training_pairs-{}.pkl'.format(self.run_id), 'wb'))
        
        # Tune the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=training_epochs, 
                  evaluation_steps=evaluation_steps, 
                  warmup_steps=100,
                  output_path=self.output_dir + '/aa_model-{}'.format(self.run_id),
                  checkpoint_path=self.output_dir + '/aa_model_{}_checkpoints/'.format(self.run_id),
                  checkpoint_save_steps=evaluation_steps,
                  checkpoint_save_total_limit=3,
                  evaluator=eer_valid_evaluator)

        

        print('EER after training', self.model.evaluate(eer_test_evaluator))


    def predict_and_evaluate(self, output_path, input_path=None, batch_size=16, with_filtering=False, filter_prcnt=0.5):
        
        if input_path != None:
            queries_fname, candidates_fname = PAUSIT_MODEL.get_file_names(input_path + '/data')
            query_df = pd.read_json(queries_fname, lines=True)
            candidates_df = pd.read_json(candidates_fname, lines=True)
            query_df = PAUSIT_MODEL.modify_df(data=query_df, author_identifier='authorIDs', ratio=4)
            
            candidates_df['authorSetID'] = candidates_df.authorSetIDs.apply(lambda x: x[0][2:-3] if x[0].startswith("(") else x[0])
            query_df['authorID'] = query_df.authorIDs.apply(lambda x: x[0][2:-3] if x[0].startswith("(") else x[0])

            ground_truth_path = input_path + '/groundtruth'
            ground_truth_assignment = np.load(open(ground_truth_path + '/hrs_release_08-14-23_crossGenre-combined_TA2_groundtruth.npy', 'rb'))
            candidate_authors = [a[2:-3] for a in  open(ground_truth_path + '/hrs_release_08-14-23_crossGenre-combined_TA2_candidate-labels.txt').read().split('\n')][:-1]
            query_authors = [a[2:-3] for a in  open(ground_truth_path + '/hrs_release_08-14-23_crossGenre-combined_TA2_query-labels.txt').read().split('\n')][:-1]


        else:
            query_df = self.query_df
            candidates_df = self.candidates_df
            ground_truth_assignment = np.load(open(self.valid_ground_truth_path + self.prefix + '_TA2_groundtruth.npy', 'rb'))
            candidate_authors = [a[2:-3] for a in  open(self.valid_ground_truth_path + self.prefix + '_TA2_candidate-labels.txt').read().split('\n')][:-1]
            query_authors = [a[2:-3] for a in  open(self.valid_ground_truth_path + self.prefix + '_TA2_query-labels.txt').read().split('\n')][:-1]


        eer_evaluator = EER_Evaluator(query_df, candidates_df, 
            ground_truth_assignment, query_authors, candidate_authors, batch_size=batch_size, as_predictor=True, with_filtering=with_filtering, mask=mask)


        eer_metric, auc_metric, author_pairwise_sim, query_authors, candidate_authors = self.model.evaluate(eer_evaluator)
        
        prefix = PAUSIT_MODEL.get_prefix(input_dir=self.input_dir)
        PAUSIT_MODEL.save_output_files(prefix, output_path, self.run_id, author_pairwise_sim, query_authors, candidate_authors)

        return eer_metric, auc_metric
            

    def predict(self, batch_size=32, with_filtering=False):
        #self.query_df = self.query_df.sample(10)
        #self.candidates_df = self.candidates_df.sample(100)

        self.query_df['authorID'] = self.query_df.authorIDs.apply(lambda x : x[0])
        self.candidates_df['authorSetID'] = self.candidates_df.authorSetIDs.apply(lambda x : x[0])

        author_pairwise_sim, query_authors, candidate_authors = compute_pairwise_selection(self.model, self.query_df, self.candidates_df)

        prefix = PAUSIT_MODEL.get_prefix(input_dir=self.input_dir)
        PAUSIT_MODEL.save_output_files(prefix, self.output_dir, self.run_id, author_pairwise_sim, query_authors, candidate_authors)