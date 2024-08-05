import os
import argparse
import spacy

import pandas as pd
from mutual_implication_score import MIS

from utils import replace_first_word, get_np
from styles import StyleGenerator, process_style_features, refine_features


def main(args):
    """
    Main function for dynamically creating style descriptions (and filtering).
    Implements a pipeline for generating and filtering writing style descriptions from text documents.

    Parameters:
    args (dict): Dictionary of arguments including:
        --device: List of device IDs for computation (e.g., [0] for a single GPU).
        --generator-model: Name of the model to use for generating styles (default: "llama3").
        --shortener-model: Name of the model to use for shortening styles (default: "llama3").
        --max-new-tokens: Maximum number of new tokens to generate (default: 512).
        --data-dir: Directory containing the input data in JSONL format. Should look something like:
          ```
            {"documentID":"[DOC_ID]","authorIDs":["[AUTHOR_ID]"],"fullText":"[TEXT]","languages":["en"],"lengthWords":[WORD_LEN]}
          ```
        --filter-pipeline: String specifying the filter pipeline to use.

    Steps:
    --------------
    1. Describe Styles: Use LLM to describe the style of each document (e.g., "The author uses X").
    2. Distill Corpus of Style Features:
        - Filter out features occurring in fewer than some number of documents.
        - Shorten remaining features.
    3. MIS Clustering: Perform MIS clustering to remove duplicates.
    4. Extract Noun Phrases: Extract noun phrases from each sentence.

    Output:
    - The generated and processed data is saved in CSV format at various stages:
    - Initial style descriptions:      describe_documents_writing_styles.csv
    - Shortened style descriptions:    style_corpus.csv
    - Filtered style descriptions:     filtered/style_corpus.csv
    - Refined and aggregated features: filtered/refined_and_aggregated_features.csv
    - Final extracted phrases:         filtered/refined_and_aggregated_features_final.csv

    Example usage:
    --------------
    python generate_styles.py --device 0 --generator-model llama3 --max-new-tokens 512 --data-dir ./data/input.jsonl
    """
    # Load data
    training_df = pd.read_json(args["data_dir"], lines=True)
    style_generator = StyleGenerator(
        model_name=args["generator_model"],
        device=args["device"],
        max_new_tokens=args["max_new_tokens"],
    )

    print("Generating styles... ", end="")
    save_dir = (
        replace_first_word(args["data_dir"], "describe_documents_writing_styles")
        + ".csv"
    )

    if not os.path.exists(save_dir):
        output_df = style_generator.describe_documents_writing_styles(
            training_df.documentID.tolist(), training_df.fullText.tolist()
        )

        # Extract styles and map with documentIDs
        styles_df = pd.DataFrame(columns=["attribute_name", "documentID"])

        for _, row in output_df.iterrows():
            descriptions = [
                i.replace("* ", "")
                for i in row["generations"].split("\n\n")
                if "**" not in i and "*" in i
            ]
            
            if len(descriptions) > 0:
                descriptions = descriptions[0].split("\n")
            else:
                print(descriptions)
                continue;

            for description in descriptions:
                styles_df = pd.concat(
                    [
                        styles_df,
                        pd.DataFrame(
                            [[description, row["documentID"]]],
                            columns=["attribute_name", "documentID"],
                        ),
                    ]
                )
        styles_df.to_csv(save_dir, index=False)
    else:
        styles_df = pd.read_csv(save_dir)
    print("Done.")

    print("Shortening styles... ", end="")
    save_dir = os.path.join(os.path.dirname(args["data_dir"]), "style_corpus.csv")

    style_shortener = StyleGenerator(
        model_name=args["shortener_model"],
        device=args["device"],
        max_new_tokens=args["max_new_tokens"],
    )

    if not os.path.exists(save_dir):
        # Shorten generated style descriptions
        style_corpus = process_style_features(
            style_shortener,
            styles_df,
            datadreamer_path="processing_llama3_style_features",
            column_name="attribute_name",
            output_clm="shortend_attribute_name.v1",
            output_path=save_dir,
        )

        # Another round of shortening
        style_corpus = process_style_features(
            style_shortener,
            style_corpus,
            datadreamer_path="processing_llama3_style_features.v2",
            column_name="shortend_attribute_name.v1",
            output_clm="shortend_attribute_name.v2",
            output_path=save_dir,
        )
    else:
        style_corpus = pd.read_csv(save_dir)
    print("Done.")

    print("Filtering shortened styles... ", end="")
    save_dir = os.path.join(
        os.path.dirname(args["data_dir"]), "filtered", "style_corpus.csv"
    )

    if not os.path.exists(save_dir):
        feat_to_document_df = (
            style_corpus.groupby("shortend_attribute_name.v2")
            .agg({"documentID": lambda x: len(list(x))})
            .reset_index()
        )
        # Keep features with minimum frequency
        feats_with_3_minimum_freq = feat_to_document_df[
            feat_to_document_df.documentID > args["style_threshold"]
        ]["shortend_attribute_name.v2"].tolist()
        filtered_df = style_corpus[
            style_corpus["shortend_attribute_name.v2"].isin(feats_with_3_minimum_freq)
        ]

        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        filtered_df.to_csv(save_dir, index=False)
    else:
        filtered_df = pd.read_csv(save_dir)
    print("Done.")

    save_dir = os.path.join(
        os.path.dirname(args["data_dir"]),
        "filtered",
        "refined_and_aggregated_features.csv",
    )

    if not os.path.exists(save_dir):
        mis = MIS(device=f'cuda:{args["device"][0]}')

        filtered_df["shortend_attribute_name.v2"] = filtered_df[
            "shortend_attribute_name.v2"
        ].apply(lambda x: x.split("\n")[0].strip())

        unique_feats = filtered_df["shortend_attribute_name.v2"].unique()
        refined_feats = refine_features(unique_feats, mis)
        refine_features_dict = {item: c[0] for c in refined_feats for item in c}
        filtered_df["aggregated_name"] = filtered_df[
            "shortend_attribute_name.v2"
        ].apply(lambda x: refine_features_dict[x] if x in refine_features_dict else x)

        unique_feats = filtered_df["aggregated_name"].unique()
        refined_feats = refine_features(unique_feats, mis)
        refine_features_dict = {item: c[0] for c in refined_feats for item in c}
        filtered_df["aggregated_name"] = filtered_df[
            "shortend_attribute_name.v2"
        ].apply(lambda x: refine_features_dict[x] if x in refine_features_dict else x)

        filtered_df.to_csv(save_dir, index=False)
    else:
        filtered_df = pd.read_csv(save_dir)

    save_dir = os.path.join(
        os.path.dirname(args["data_dir"]),
        "filtered",
        "refined_and_aggregated_features_final.csv",
    )

    if not os.path.exists(save_dir):
        unique_attributes = filtered_df.aggregated_name.unique()
        attribute_to_patterns = {x: get_np(x) for x in unique_attributes}
        filtered_df["extracted-phrases"] = filtered_df.aggregated_name.apply(
            lambda x: attribute_to_patterns[x]
        )
        expanded_df = pd.DataFrame(
            [
                (
                    x,
                    row["attribute_name"],
                    row["documentID"],
                    row["shortend_attribute_name.v1"],
                    row["shortend_attribute_name.v2"],
                    row["aggregated_name"],
                )
                for _, row in filtered_df.iterrows()
                for x in row["extracted-phrases"]
            ],
            columns=[
                "final_attribute_name",
                "original_attribute_name",
                "documentID",
                "shortend_attribute_name.v1",
                "shortend_attribute_name.v2",
                "aggregated_name",
            ],
        )

        expanded_df.to_csv(save_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--device", type=int, nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--generator-model",
        type=str,
        default="llama3",
        choices=[
            "llama3",
            "mistral",
            "openai:gpt-3.5-turbo",
            "openai:gpt-4",
            "openai:gpt-4o",
        ],
    )
    parser.add_argument(
        "--shortener-model",
        type=str,
        default="llama3",
        choices=[
            "llama3",
            "mistral",
            "openai:gpt-3.5-turbo",
            "openai:gpt-4",
            "openai:gpt-4o",
        ],
    )
    parser.add_argument("--style-threshold", type=int, default=2)
    args = vars(parser.parse_args())
    main(args)
