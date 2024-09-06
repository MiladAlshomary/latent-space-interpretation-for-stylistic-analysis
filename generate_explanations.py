import os
import argparse
import warnings
import pickle as pkl
from collections import Counter, defaultdict
import spacy

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from data import get_aa_data
from utils import find_first_minimal_change, safe_parse
from metrics import compute_model_performance
from aa_models import get_model
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


def main(args):
    """
    Main function to generate explanations for documents based on a given cluster-based interpretable space

    Parameters:
    args (dict): Dictionary of arguments including:
        --inter-space (str): path to the dataframe representing the interpretable space.
        --input-path (str): Path to the documents dataframe
        --output-path (str): Path to the output
        --top-k (int): Number of top features to take from the cluster
        --top-c (int): Number of top dimensions to take

    Steps:
    ------
    1. Generate the latent embeddings of all documents
    2. Project documents using the corresponding proj_matrix of the interpretable space
    3. Take the top-k features of the top-c dimensions (clusters) as the style explanations of the document 
    
    Example usage:
    --------------
    python generate_explanations.py --inter-space <path to the interpretable space generated from cluster_documents.py code> 
                                    --input-path <path to the dataframe containing the documents to be explained> 
                                    --output-path <path to where to save the output>
    
    """

    model = get_model(args["model"])

    
    # Load model and data
    #'../data/explainability/explainability_experiment_test_ds.pkl'
    documents_df = pd.read_json(args['input_path'])
    
    #'../data/explainability/clusterd_authors_with_style_description.pkl'
    interpretable_space = pkl.load(open(args['inter_path'], 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    dimension_to_style  = {key: interpretable_space[key][1] for key in interpretable_space}

    proj_matrix = np.array(list(dimension_to_latent.values()))
    #print(proj_matrix)
    #normalize projection matrix
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    documents_assigned_clusters, documents_ranked_clusters = document_to_cluster_assignment(model, proj_matrix, documents_df.fullText.tolist())

    # Aggregate the top-k fetures of the top-n clusters to be the final list
    final_documents_reps = []
    for i, ranked_clusters in enumerate(documents_ranked_clusters):
        rep_feats = []
        for cluster_id in ranked_clusters[:args['top_c']]:
            cluster_feats = sorted(dimension_to_style[cluster_id].items(), key=lambda x: -x[1])
            #print(cluster_id)
            #print(cluster_feats)
            rep_feats+= [x[0] for x in cluster_feats[:args['top_k']]]
            #break
        final_documents_reps.append(rep_feats)
        #break

    documents_df['documetn_style_description'] = final_documents_reps
    documents_df.to_json(args['output_path'])


    
def document_to_cluster_assignment(model, proj_matrix, documents):

    documents_latent = model.encode(documents)
    documents_interp  = [proj_matrix.dot(e) for e in documents_latent]

    cluster_scores  = [x.tolist() for x in documents_interp]
    ranked_clusters = [list(np.argsort(x)[::-1]) for x in cluster_scores]

    return cluster_scores, ranked_clusters
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--inter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-c", type=int, required=True)
    args = vars(parser.parse_args())
    print(args)
    main(args)
