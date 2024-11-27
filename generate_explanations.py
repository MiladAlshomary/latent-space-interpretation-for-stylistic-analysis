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
from sklearn.metrics.pairwise import cosine_similarity


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
        final_documents_reps.append(rep_feats)

    documents_df['documetn_style_description'] = final_documents_reps
    documents_df.to_json(args['output_path'])

def get_documents_style_descriptions(documents, model_path, interp_space_path, interp_space_rep_path, style_feat_clm, top_c=3, top_k=10, flip_cluster_order=False):
    
    model = get_model(model_path)
    
    #'../data/explainability/clusterd_authors_with_style_description.pkl'
    interpretable_space = pkl.load(open(interp_space_path, 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    
    #Load interp space representations
    interpretable_space_rep_df = pd.read_json(interp_space_rep_path)
    dimension_to_style  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}

    
    proj_matrix = np.array(list(dimension_to_latent.values()))
    #print(proj_matrix)
    #normalize projection matrix
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')
    
    documents_assigned_clusters, documents_ranked_clusters = document_to_cluster_assignment(model, proj_matrix, documents)

    #print([r[0] for r in documents_ranked_clusters])
    
    # Aggregate the top-k fetures of the top-n clusters to be the final list
    final_documents_reps = []
    final_documents_clusters = []
    final_documents_dist_rep = []
    for i, ranked_clusters in enumerate(documents_ranked_clusters):
        rep_feats = []
        if flip_cluster_order:
            ranked_clusters = list(reversed(ranked_clusters))

        for cluster_id in ranked_clusters[:top_c]:
            cluster_feats = dimension_to_style[cluster_id]
            rep_feats+= [(cluster_id, x) for x in cluster_feats[:top_k]]
        final_documents_reps.append(rep_feats)
        final_documents_clusters.append([(c, documents_assigned_clusters[i][c]) for c in ranked_clusters[:top_c]])
    
    return final_documents_reps, final_documents_clusters


def get_documents_rep_vectors(documents, model_path, interp_space_path):
    model = get_model(model_path)
    
    interpretable_space = pkl.load(open(interp_space_path, 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    print("# clusters:", len(interpretable_space))

    
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    proj_matrix = np.array(list(dimension_to_latent.values()))
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    # Compute latent and interpretable vectors for query and candidate documents
    documents_latent = model.encode(documents)
    documents_interp  = [proj_matrix.dot(e) for e in documents_latent]

    return documents_latent, documents_interp
    
def explain_model_prediction(model_path, inter_space_path, query_document, candidate_documents, top_c=3, top_k=5):

    #load ta2 model
    model = get_model(model_path)
    
    #load interpretable_space and compute projection matrix
    interpretable_space = pkl.load(open(inter_space_path, 'rb'))
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    dimension_to_style  = {key: [f[0] for f in sorted(interpretable_space[key][1].items(), key=lambda x: -x[1])] for key in interpretable_space}
    dimension_to_style_summary  = {key: interpretable_space[key][2] for key in interpretable_space}
    proj_matrix = np.array(list(dimension_to_latent.values()))
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    # Compute latent and interpretable vectors for query and candidate documents
    documents_latent = model.encode([query_document] + candidate_documents)
    documents_interp  = [proj_matrix.dot(e) for e in documents_latent]

    # Compute Model's latent and interp prediction
    latent_similarities = cosine_similarity(documents_latent[:1], documents_latent[1:])
    interp_similarities = cosine_similarity(documents_interp[:1], documents_interp[1:])
    model_latent_rank = np.argsort(latent_similarities[0])[::-1]
    model_interp_rank = np.argsort(interp_similarities[0])[::-1]

    #### Explanation #####

    # Find cluster assignment for the query and candidate documents
    query_cluster_assignments = documents_interp[0].tolist()
    query_cluster_rankings    = np.argsort(query_cluster_assignments)[::-1]

    candidate_cluster_assignments = [interp.tolist() for interp in documents_interp[1:]]
    candidate_cluster_rankings    = [np.argsort(ass)[::-1] for ass in candidate_cluster_assignments]

    # Extract the style descriptions of top_c clusters similar to the query document
    query_style_reps = [dimension_to_style[cidx][:top_k] for cidx in query_cluster_rankings[:top_c]]
    query_style_reps_summ = [dimension_to_style_summary[cidx] for cidx in query_cluster_rankings[:top_c]]

    # Compute how similar the candidate documents to these top_c clusters
    candidates_distance_to_query_rep = [[c_assignment[cidx] for cidx in query_cluster_rankings[:top_c]]
        for c_assignment in candidate_cluster_assignments]
    
    #candidate_query_cluster_overlap  = [set(query_cluster_rankings[:top_c]).intersection(set(x[:top_c])) for x in candidate_cluster_rankings]
    #candidates_distance_to_query_rep = [[c_assignment[cidx] for cidx in query_cluster_rankings[:top_c]]
    #    for c_assignment in candidate_cluster_assignments]

    # candidates_similarity_to_query = [
    #     {
    #         'num_shared_clusters': len(shared_clusters),
    #         'shared_clusters': shared_clusters,
    #         'dist_to_query_rep': candidates_distance_to_query_rep[i],
    #         'shared_feats': [f for c_id in shared_clusters for f in dimension_to_style[c_id][:top_k]]
    #     }
    #     for i, shared_clusters in enumerate(candidate_query_cluster_overlap)
    # ]
    
    
    return model_latent_rank, model_interp_rank, query_style_reps, query_style_reps_summ, candidates_distance_to_query_rep

def explain_model_prediction_over_author(model_path, inter_space_path, inter_space_rep_path, query_author, candidate_authors, top_c=3, top_k=5, style_feat_clm='tfidf_rep_5', style_feat_summary_clm=None):

    #load ta2 model
    model = get_model(model_path)
    
    #load interpretable_space and compute projection matrix
    interpretable_space = pkl.load(open(inter_space_path, 'rb'))

    #Load interp space representations
    interpretable_space_rep_df = pd.read_json(inter_space_rep_path)
    dimension_to_style  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}

    if style_feat_summary_clm == None:
        dimension_to_style_summary  = {x[0]: ' - '.join(x[1]) for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_clm].tolist())}
    else:
        dimension_to_style_summary  = {x[0]: x[1] for x in zip(interpretable_space_rep_df.cluster_label.tolist(), interpretable_space_rep_df[style_feat_summary_clm].tolist())}
    
    del interpretable_space[-1] #DBSCAN generate a cluster -1 of all outliers. We don't want this cluster
    print("# clusters:", len(interpretable_space))
    dimension_to_latent = {key: interpretable_space[key][0] for key in interpretable_space}
    
    proj_matrix = np.array(list(dimension_to_latent.values()))
    proj_matrix = normalize(proj_matrix, axis=1, norm='l2')

    # Compute latent and interpretable vectors for query and candidate authors
    q_author_latents = model.encode(query_author)
    q_author_interps = [proj_matrix.dot(e) for e in q_author_latents]
    
    c_author_latents = [model.encode(c_author) for c_author in candidate_authors]
    c_author_interps = [[proj_matrix.dot(e) for e in documents_latent] for documents_latent in c_author_latents]

    # Author representations as an average of their documents
    q_author_latent_avg  = np.mean(q_author_latents, axis=0)
    q_author_interp_avg  = np.mean(q_author_interps, axis=0)
    c_author_latent_avgs = [np.mean(x, axis=0) for x in c_author_latents]
    c_author_interp_avgs = [np.mean(x, axis=0) for x in c_author_interps]
    
    # Compute Model's latent and interp prediction
    latent_similarities = [np.mean(cosine_similarity(q_author_latents, c_latent)) for c_latent in c_author_latents]
    interp_similarities = [np.mean(cosine_similarity(q_author_interps, c_interp)) for c_interp in c_author_interps]
    
    model_latent_rank = np.argsort(latent_similarities)[::-1]
    model_interp_rank = np.argsort(interp_similarities)[::-1]

    #### Explanation #########

    # Find cluster assignment for the query and candidate documents
    query_cluster_assignments = q_author_interp_avg.tolist()
    query_cluster_rankings    = np.argsort(query_cluster_assignments)[::-1]

    candidate_cluster_assignments = [interp.tolist() for interp in c_author_interp_avgs]
    candidate_cluster_rankings    = [np.argsort(ass)[::-1] for ass in candidate_cluster_assignments]

    # Extract the style descriptions of top_c clusters similar to the query document
    query_style_reps = [dimension_to_style[cidx][:top_k] for cidx in query_cluster_rankings[:top_c]]
    query_style_reps_summ = [dimension_to_style_summary[cidx] for cidx in query_cluster_rankings[:top_c]]

    # Compute how similar the candidate documents to these top_c clusters
    candidates_distance_to_query_rep = [[c_assignment[cidx] for cidx in query_cluster_rankings[:top_c]]
        for c_assignment in candidate_cluster_assignments]

    
    return model_latent_rank, model_interp_rank, query_style_reps, query_style_reps_summ, candidates_distance_to_query_rep
    
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
