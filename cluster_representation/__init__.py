import sys

from contra_sum import contra_sum_utils, subm
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import math
from collections import Counter, defaultdict


def get_style_feats_distribution(documentIDs, style_feats_dict):
    style_feats = []
    for documentId in documentIDs:
        if documentId not in document_to_style_feats:
            #print(documentId)
            continue

        style_feats+= document_to_style_feats[documentId]

    #print(len(style_feats))
    tfidf = [style_feats.count(key) * val for key, val in style_feats_dict.items()]
    #print(len(tfidf))
    #print(tfidf)
    return tfidf

def get_cluster_top_feats(style_feats_distribution, style_feats_list, top_k=5):
    sorted_feats = np.argsort(style_feats_distribution)[::-1]
    #print(sorted(list(zip(style_feats_list, style_feats_distribution)), key=lambda x: -x[1])[:10])
    #print([style_feats_list[x] for x in sorted_feats[:top_k]])
    #print([style_feats_distribution[i] for i in sorted_feats[:top_k]])
    top_feats = [style_feats_list[x] for x in sorted_feats[:top_k] if style_feats_distribution[x] > 0]
    return top_feats

def get_contrastive_cluster_representation(style_features, labels, k=5, la=0.9, verbose=False):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    kernel = contra_sum_utils.Kernel.create(model.encode(style_features), metric = 'cosine')
    
    V = [i for i, f in enumerate(style_features)]
    
    #representativeness and contrastiveness
    sol_greedy = subm.greedy_maximize_labels(subm.MMD(), -1.0 * subm.MMD(), 
                                         V = V, y = labels, k = k, lambdaa=la, verbose=verbose,
                                         delF_args = {"K": kernel}, delG_args = {"K": kernel})

    
    style_feats_and_labels = list(zip(style_features, labels))
    return  {x[0]:[style_feats_and_labels[y] for y in x[1]] for x in sol_greedy.items()}
    
def average_overlap(desc1, desc2):
    desc1 = ' '.join(desc1)# if type(desc1) == list else desc1
    desc2 = ' '.join(desc2)# if type(desc1) == list else desc2
    
    desc1 = set(nltk.word_tokenize(desc1.lower()))
    desc2 = set(nltk.word_tokenize(desc2.lower()))
    return len(desc1.intersection(desc2))/len(desc1.union(desc2))


def generate_interpretable_space_contra_representation(interp_space_path, styles_df_path, feat_clm, output_clm, num_feats=5, summarize_with_gpt=False):

    styles_df         = pd.read_csv(styles_df_path)[[feat_clm, "documentID"]]
    doc_style_agg_df  = styles_df.groupby('documentID').agg({feat_clm: lambda x : list(x)}).reset_index()
    document_to_feats_dict = {x[0]: x[1] for x in zip(doc_style_agg_df.documentID.tolist(), doc_style_agg_df[feat_clm].tolist())}
    
    df = pd.read_pickle(interp_space_path)
    df = df[df.cluster_label != -1]
    clusterd_df = df.groupby('cluster_label').agg({
        'documentID': lambda x: [d_id for doc_ids in x for d_id in doc_ids]
    }).reset_index()

    #Filter-in only documents that has style description
    clusterd_df['documentID'] = clusterd_df.documentID.apply(lambda documentIDs: [documentID for documentID in documentIDs if documentID in document_to_feats_dict])
    
    clusterd_df[feat_clm] = clusterd_df.documentID.apply(lambda doc_ids: list(set([f for d_id in doc_ids for f in document_to_feats_dict[d_id]])))

    cluster_partition = [[0]] + [[label] for label in clusterd_df.cluster_label]
    cluster_feats = [(p_id, clusterd_df[clusterd_df.cluster_label.isin(partition)][feat_clm].tolist()) for p_id, partition in enumerate(cluster_partition)]
    cluster_feats = [(f, p_id) for p_id, clusters in cluster_feats for feats in clusters for f in feats]

    style_features, labels = zip(*cluster_feats)
    labels = np.array(labels)
    solution = get_contrastive_cluster_representation(style_features, labels, k=num_feats, la=0.1, verbose=False)

    clusterd_df[output_clm] = clusterd_df.cluster_label.apply(lambda c_lable: [x[0] for x in solution[c_lable+1]])

    if summarize_with_gpt:
        clusterd_df[output_clm] = clusterd_df[output_clm].apply(lambda feats: style_generator.summarize_sentences([top_k_feats], 'to_concise_paragraph')[0])
        
    return clusterd_df

def generate_interpretable_space_representation(interp_space_path, styles_df_path, feat_clm, output_clm, num_feats=5, summarize_with_gpt=False):
    
    styles_df = pd.read_csv(styles_df_path)[[feat_clm, "documentID"]]
    
    style_feats_agg_df = styles_df.groupby(feat_clm).agg({'documentID': lambda x : len(list(x))}).reset_index()
    style_feats_agg_df['document_freq'] = style_feats_agg_df.documentID

    #create a dictionary mapping features to their idf
    style_feats_list = style_feats_agg_df[feat_clm].tolist()
    doc_style_agg_df     = styles_df.groupby('documentID').agg({feat_clm: lambda x : list(x)}).reset_index()
    document_to_feats_dict = {x[0]: x[1] for x in zip(doc_style_agg_df.documentID.tolist(), doc_style_agg_df[feat_clm].tolist())}
    number_documents     = styles_df.documentID.nunique()
    style_to_feats_dfreq = {x[0]: math.log(number_documents/x[1]) for x in zip(style_feats_agg_df[feat_clm].tolist(), style_feats_agg_df.document_freq.tolist())}

    df = pd.read_pickle(interp_space_path)
    df = df[df.cluster_label != -1]
    clusterd_df = df.groupby('cluster_label').agg({
        'documentID': lambda x: [d_id for doc_ids in x for d_id in doc_ids]
    }).reset_index()

    #Filter-in only documents that has a style description
    clusterd_df['documentID'] = clusterd_df.documentID.apply(lambda documentIDs: [documentID for documentID in documentIDs if documentID in document_to_feats_dict])
    
    clusterd_df[feat_clm] = clusterd_df.documentID.apply(lambda doc_ids: [f for d_id in doc_ids for f in document_to_feats_dict[d_id]])

    def compute_tfidf(row):
        style_counts = Counter(row[feat_clm])
        total_num_styles = sum(style_counts.values())
        #print(style_counts, total_num_styles)
        style_distribution = {
            style: math.log(1+count) * style_to_feats_dfreq[style] if style in style_to_feats_dfreq else 0 for style, count in style_counts.items()
        } #TF-IDF
        
        return style_distribution

    def create_tfidf_rep(tfidf_dist, num_feats, summarize_with_gpt):
        style_feats = sorted(tfidf_dist.items(), key=lambda x: -x[1])
        top_k_feats = [x[0] for x in style_feats[:num_feats] if str(x[0]) != 'nan']
        if summarize_with_gpt:
            rep_summary_df = style_generator.summarize_sentences([top_k_feats], 'to_concise_paragraph')
            rep_summary = rep_summary_df.generations.tolist()
            return rep_summary[0]
        else:
            return top_k_feats

    clusterd_df[output_clm +'_dist'] = clusterd_df.apply(lambda row: compute_tfidf(row), axis=1)
    clusterd_df[output_clm]         = clusterd_df[output_clm +'_dist'].apply(lambda dist: create_tfidf_rep(dist, num_feats, summarize_with_gpt))

    return clusterd_df
    
    