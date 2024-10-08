import os
import argparse
import warnings
import json
import pickle as pkl
from collections import Counter, defaultdict
import spacy
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from data import get_aa_data
from utils import find_first_minimal_change, safe_parse
from metrics import compute_model_performance
from aa_models import get_model
from styles import StyleGenerator

warnings.filterwarnings("ignore")


def main(args):
    """
    Main function for training and testing AA model. Uses DBSCAN for clustering.
    Computes performance for different epsilon values. Saves best results.

    Parameters:
    args (dict): Dictionary of arguments including:
        --train-dir (str): Directory for training data.
        --test-dir (str): Directory for testing data.
        --save-dir (str): Directory to save results.
        --model (str): Model name to use.
        --eps-threshold (float): Threshold for epsilon performance change.

    Steps:
    ------
    1. Load model and data.
    2. Get embeddings for documents.
    3. Compute dissimilarity matrix.
    4. Loop through epsilon values. Compute performance.
    5. Find best epsilon. Label data.
    6. Save results as .pkl file.

    Example usage:
    --------------
    python script.py --train-dir path/to/train --test-dir path/to/test --save-dir path/to/save --model aa_model-luar --eps
    """
    # Load model and data
    model = get_model(args["model"])
    train_df = get_aa_data(args["train_dir"], group_by_author=True)
    test_df  = get_aa_data(args["test_dir"])

    # Get embeddings for documents
    author_to_embeddings = {
        row["authorID"]: np.mean(model.encode(row["fullText"]), axis=0)
        for _, row in train_df.iterrows()
    }
    author_mean_embeddings = np.vstack(list(author_to_embeddings.values()))

    # Compute dissimilarity matrix
    distance_matrix = squareform(pdist(author_mean_embeddings, metric="cosine"))

    def find_best_eps():
        # Loop through a range of epsilons and compute their performance
        eps_to_performance = {}
        eps_to_labels = {}
        
        for eps in tqdm(
            np.arange(0.01, 1, 0.01),
            ascii=True,
            desc="Testing Different Epsilon Values",
            leave=False,
        ):
            cluster_labels = DBSCAN(
                eps=eps, metric="precomputed", min_samples=2, n_jobs=-1
            ).fit_predict(distance_matrix)
    
            sum_vectors = defaultdict(lambda: np.zeros(len(author_mean_embeddings[0])))
            count_vectors = Counter(cluster_labels)
            for label, vector in zip(cluster_labels, author_mean_embeddings):
                sum_vectors[label] += vector
    
            new_bases = np.array(
                [sum_vectors[label] / count for label, count in count_vectors.items()]
            )
    
            eps_to_performance[eps] = compute_model_performance(model, test_df, new_bases)
            eps_to_performance[eps] = [eps_to_performance[eps], len(new_bases)]
            eps_to_labels[eps] = cluster_labels
    
            print(eps_to_performance[eps], eps)

        #json.dump(eps_to_performance, open('./eps_performances.json', 'w'))
        best_eps = sorted(eps_to_performance.items(), key=lambda x: x[1][0])[0][0]#find_first_minimal_change(eps_to_performance, args["eps_threshold"])
        train_df["cluster_label"] = eps_to_labels[best_eps]

        return best_eps, train_df

    if args['eps'] == -1:
        best_eps, train_df = find_best_eps()
    else:
        best_eps = args['eps']

        cluster_labels = DBSCAN(
                eps=best_eps, metric="precomputed", min_samples=2, n_jobs=-1
            ).fit_predict(distance_matrix)
    
        sum_vectors = defaultdict(lambda: np.zeros(len(author_mean_embeddings[0])))
        count_vectors = Counter(cluster_labels)
        for label, vector in zip(cluster_labels, author_mean_embeddings):
            sum_vectors[label] += vector

        new_bases = np.array(
            [sum_vectors[label] / count for label, count in count_vectors.items()]
        )

        train_df["cluster_label"] = cluster_labels
        
        
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    styles_df = pd.read_csv(args["style_dir"])[["final_attribute_name", "documentID"]]

    train_df["documentID"] = train_df["documentID"].apply(safe_parse)
    train_df_exploded = train_df.explode("documentID")
    train_df_exploded["fullText"] = train_df_exploded["fullText"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else x
    )
    merged_df = pd.merge(train_df_exploded, styles_df, on="documentID", how="left")
    aggregated_df = (
        merged_df.groupby(["authorID", "fullText", "documentID", "cluster_label"])
        .agg({"final_attribute_name": lambda x: x.tolist()})
        .reset_index()
    )
    final_agg = (
        aggregated_df.groupby(["authorID", "fullText", "cluster_label"])
        .agg({"final_attribute_name": lambda x: sum(x, [])})
        .reset_index()
    )
    train_df["fullText"] = train_df["fullText"].apply(
        lambda x: " ".join(x) if isinstance(x, list) else x
    )
    final_df = pd.merge(
        train_df, final_agg, on=["authorID", "fullText", "cluster_label"], how="left"
    )

    # Save clustered and style-assigned author DataFrame
    pd.to_pickle(
        final_df,
        open(
            os.path.join(
                args["save_dir"],
                os.path.basename(os.path.splitext(args["train_dir"])[0] + ".pkl"),
            ),
            "wb",
        ),
    )

    # Construct interpretable dimensions to style distribution mapping
    cluster_to_authors = defaultdict(list)
    cluster_to_styles = defaultdict(list)

    #create a dictionary mapping features to their idf
    number_documents    = styles_df.documentID.nunique()
    style_feats_agg_df  = styles_df.groupby('final_attribute_name').agg({'documentID': lambda x: len(x)}).reset_index()
    style_feats_agg_df['document_freq'] = style_feats_agg_df.documentID
    style_feats_list = style_feats_agg_df.final_attribute_name.tolist()
    style_to_feats_dfreq = {x[0]: math.log(number_documents/x[1]) for x in zip(style_feats_agg_df.final_attribute_name.tolist(), style_feats_agg_df.document_freq.tolist())}

    for _, row in final_df.iterrows():
        cluster_label = row["cluster_label"]
        author_id = row["authorID"]
        styles = row["final_attribute_name"]

        cluster_to_authors[cluster_label].append(author_to_embeddings[author_id])
        cluster_to_styles[cluster_label].extend(styles)

    cluster_to_avg_vector = {}
    for cluster_label, vectors in cluster_to_authors.items():
        avg_vector = np.mean(vectors, axis=0)
        cluster_to_avg_vector[cluster_label] = avg_vector

    vector_to_style_distribution = {}
    for cluster_label, avg_vector in cluster_to_avg_vector.items():
        style_counts = Counter(cluster_to_styles[cluster_label])
        total_styles = sum(style_counts.values())
        style_distribution = {
            style: count * style_to_feats_dfreq[style] if style in style_to_feats_dfreq else 0 for style, count in style_counts.items()
        } #TF-IDF
        
        vector_to_style_distribution[cluster_label] = [tuple(avg_vector), style_distribution]

    if args['summarize_cluster_reps']:
        print('Summarizing styles of interpretable dimensions')
        style_generator = StyleGenerator(model_name="openai:gpt-3.5-turbo", device=0, max_new_tokens=100)
        for key, value in vector_to_style_distribution.items():
            style_feats_dict = value[1]
            style_feats = sorted(style_feats_dict.items(), key=lambda x: -x[1])
            print(style_feats[:args['top_k_feats']])
            top_k_feats = [x[0] for x in style_feats[:args['top_k_feats']] if str(x[0]) != 'nan']
            rep_summary_df = style_generator.summarize_sentences([top_k_feats], 'to_concise_paragraph')
            rep_summary = rep_summary_df.generations.tolist()
            print(top_k_feats)
            print(rep_summary)
            vector_to_style_distribution[key] = (value[0], value[1], rep_summary[0])
            
    # Save final interpretable space
    pkl.dump(
        vector_to_style_distribution,
        open(os.path.join(args["save_dir"], "interpretable_space.pkl"), "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--style-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="aa_model-luar")
    
    parser.add_argument("--eps", type=float, default=-1)
    parser.add_argument("--summarize_cluster_reps", action='store_true', default=False)
    parser.add_argument("--top_k_feats", type=int, default=-1)
    args = vars(parser.parse_args())

    if not os.path.exists(args["style_dir"]):
        raise AssertionError("Please run `generate_styles.py` before clustering.")
    main(args)
