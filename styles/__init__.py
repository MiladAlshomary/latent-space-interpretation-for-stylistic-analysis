import os
import re
import sys
import json
import itertools
import pickle as pkl
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from functools import partial
from huggingface_hub import login
from sklearn.cluster import DBSCAN
from datadreamer import DataDreamer
from datadreamer.steps import ProcessWithPrompt, DataSource, zipped, concat

from utils import (
    gen_from_iterable_dataset,
    extract_feats,
    word_difference,
    column,
    merge_sublists,
    is_sentence_finished
)
from backbones import get_datadreamer_backbone


############################
# DOCUMENT GENERATOR CLASS #
############################
class StyleGenerator:
    """
    A class used to generate and manipulate styled documents using a specified model.

    Attributes:
        model_name (str): Name of the model.
        max_new_tokens (int): Maximum number of new tokens to generate.
        device (str, optional): Device to run the model on. Defaults to None.
        taxonomy_name (str, optional): Name of the taxonomy to use. Defaults to None.
        datadreamer_path (str): Path to DataDreamer directory. Defaults to ".datadreamer".
        style_features_folder (str): Folder to save style features. Defaults to "styles/".
    """

    def __init__(
        self,
        model_name,
        max_new_tokens,
        device=None,
        taxonomy_name=None,
        datadreamer_path=".datadreamer",
        style_features_folder="styles/",
    ):
        """
        Initialize the StyleGenerator with model parameters and paths.

        Parameters:
            model_name (str): Name of the model.
            max_new_tokens (int): Maximum number of new tokens to generate.
            device (str, optional): Device to run the model on. Defaults to None.
            taxonomy_name (str, optional): Name of the taxonomy to use. Defaults to None.
            datadreamer_path (str, optional): Path to DataDreamer directory. Defaults to ".datadreamer".
            style_features_folder (str, optional): Folder to save style features. Defaults to "styles/".
        """
        current_dir = os.path.dirname(__file__)

        # Load prompts for generating sentences according to styles
        self.prompts = json.load(
            open(os.path.join(current_dir, "../prompts/style_generation.json"), "r")
        )

        # Load keys
        self.keys = json.load(open(os.path.join(current_dir, "..", "keys-local.json"), "r"))

        # Authenticate HuggingFace to access gated models
        login(self.keys["huggingface"])

        self.model_name = model_name
        self.model = get_datadreamer_backbone(model_name, device)

        if taxonomy_name is not None:
            self.taxonomy_name = taxonomy_name
            style_dict = pkl.load(
                open(os.path.join(current_dir, taxonomy_name, "taxonomy.pkl"), "rb")
            )
            self.style_list = [
                style
                for style_group in list(style_dict.values())
                for style in style_group
            ]
        else:
            self.taxonomy_name = None
            self.style_list = []

        self.datadreamer_path = datadreamer_path
        self.style_features_folder = style_features_folder

        self.max_new_tokens = max_new_tokens

    def generate_styled_documents(self, num_instances, topics=None):
        """
        Generate documents with specified styles and topics.

        Parameters:
            num_instances (int): Number of instances to generate.
            topics (list, optional): List of topics to include. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing style to generated text correspondence.
        """
        save_dir = os.path.join(self.datadreamer_path, "generate")

        with DataDreamer(save_dir):
            if topics is None:
                ds_name = "style_ds"
                step_name = "Generate Documents with Styles"
                instruction = self.prompts["generate_style"]["instruction"]

                ds_list = [
                    {"t_style": style[0], "style_prompt": style[1]}
                    for style in self.style_list
                ]

                map_function = lambda row: {
                    "inputs": self.prompts["generate_style"]["style_prompt"].format(
                        row["style_prompt"].upper()
                    )
                }

            else:
                ds_name = "style_topics_ds"
                step_name = "Generate Documents with Styles and Topic"
                instruction = self.prompts["generate_style_topic"]["instruction"]

                ds_list = []
                for style, style_prompt in self.style_list:
                    ds_list.extend(
                        [
                            {"topics": x[0], "t_style": x[1], "style_prompt": x[1]}
                            for x in zip(
                                topics,
                                [style] * len(topics),
                                [style_prompt] * len(topics),
                            )
                        ]
                    )

                map_function = lambda row: {
                    "inputs": self.prompts["generate_style_topic"][
                        "style_prompt"
                    ].format(row["style_prompt"].upper(), row["topics"])
                }

            # Define the DataSource
            ds = DataSource(ds_name, Dataset.from_list(ds_list)).map(
                map_function, auto_progress=False
            )

            # Generate paragraphs
            documents_with_style = ProcessWithPrompt(
                step_name,
                inputs={"inputs": ds.output["inputs"]},
                args={
                    "llm": self.model,
                    "n": 1,
                    "max_new_tokens": self.max_new_tokens,
                    "instruction": instruction.format(num_instances),
                },
                outputs={"generations": "paragraphs"},
            ).select_columns(["paragraphs"])

            results_ds = zipped(ds, documents_with_style).output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            pandas_df.to_pickle(
                os.path.join(
                    self.style_features_folder,
                    self.taxonomy_name,
                    f"styled_documents-num_instances={num_instances}.pkl",
                )
            )

            return pandas_df

    def rewrite_documents(self, documents, rewrite_mode):
        """
        Rewrite documents with specified styles and rewrite mode.

        Parameters:
            documents (list of str): List of documents to rewrite.
            rewrite_mode (str): Mode to rewrite the documents. Can be "amplify", "reduce", or "rephrase".

        Returns:
            pandas.DataFrame: DataFrame containing the rewritten documents.
        """
        with DataDreamer(
            os.path.join(self.datadreamer_path, self.taxonomy_name, "rewrite")
        ):
            if rewrite_mode == "amplify":
                ds_name = "styles_amplify_ds"
                step_name = "Amplify Documents Style"
                instruction = self.prompts["rewrite_document"]["instruction_more"]

            elif rewrite_mode == "reduce":
                ds_name = "styles_reduce_ds"
                step_name = "Reduce Documents Style"
                instruction = self.prompts["rewrite_document"]["instruction_less"]

            elif rewrite_mode == "rephrase":
                ds_name = "styles_rephrase_ds"
                step_name = "Rephrase Documents"
                instruction = self.prompts["rewrite_document"]["instruction_rephrase"]

            else:
                raise NotImplementedError

            ds_list = []
            for style, style_prompt in self.style_list:
                ds_list.extend(
                    [
                        {"text": x[0], "t_style": x[1], "style_prompt": x[2]}
                        for x in zip(
                            documents,
                            [style] * len(documents),
                            [style_prompt] * len(documents),
                        )
                    ]
                )

            ds = DataSource(ds_name, Dataset.from_list(ds_list))

            if rewrite_mode in ["amplify", "reduce"]:
                ds = ds.map(
                    lambda row: {
                        "inputs": self.prompts["rewrite_document"][
                            "style_prompt"
                        ].format(row["style_prompt"].upper(), row["text"])
                    },
                    auto_progress=False,
                )
            else:
                ds = ds.map(lambda row: {"inputs": row["text"]}, auto_progress=False)

            documents_with_style = ProcessWithPrompt(
                step_name,
                inputs={"inputs": ds.output["inputs"]},
                args={
                    "llm": self.model,
                    "n": 1,
                    "max_new_tokens": self.max_new_tokens,
                    "instruction": instruction,
                },
                outputs={"generations": "paragraphs"},
            ).select_columns(["paragraphs"])

            results_ds = zipped(ds, documents_with_style).output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            pandas_df.to_pickle(
                os.path.join(
                    self.style_features_folder,
                    self.taxonomy_name,
                    f"rewrite_documents-mode={rewrite_mode}.pkl",
                )
            )

            return pandas_df

    def get_document_topic(self, documents, max_new_tokens=50):
        """
        Prompt model to return topics of the given documents.

        Parameters:
            documents (list of str): List of documents to get topics for.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 50.

        Returns:
            pandas.DataFrame: DataFrame containing the topics for each document.
        """
        with DataDreamer(os.path.join(self.datadreamer_path, "query_topics")):
            ds = DataSource(
                "query_topics", Dataset.from_list([{"text": x} for x in documents])
            )

            documents_topics = ProcessWithPrompt(
                "Generate Documents Topics",
                inputs={"inputs": ds.output["text"]},
                args={
                    "llm": self.model,
                    "n": 1,
                    "max_new_tokens": max_new_tokens,
                    "instruction": self.prompts["query_topics"]["instruction"],
                },
                outputs={"generations": "topics"},
            ).select_columns(["topics"])

            results_ds = documents_topics.output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            return pandas_df

    def describe_document_clusters_common_writing_styles(
        self, list_of_ids, list_of_documents, cluster_method
    ):
        """
        Describe common writing styles in document clusters.

        Parameters:
            list_of_ids (list of str): List of cluster IDs.
            list_of_documents (list of list of str): List of document lists, each corresponding to a cluster.
            cluster_method (str): Method used for clustering.

        Returns:
            pandas.DataFrame: DataFrame containing descriptions of common writing styles in each cluster.
        """

        def process_style_desc(desc):
            attrs_list = [
                re.sub(r"\(.*\)", "", f).strip()
                for f in re.findall(r"\*\*(.*)\*\*", desc)
            ]
            attrs_list += [
                re.sub(r"\(.*\)", "", f).strip()
                for f in re.findall(r"\n\*(.*)\n", desc)
            ]

            return attrs_list if len(attrs_list) > 0 else ["other"]

        if len(self.style_list) > 0:
            instruction = "Please select which of the following writing style attributes are the common attributes of the provided texts.\n"
            instruction += "Writing Style Attributes:\n{}".format(
                "\n".join([" - " + i[1].replace(".", "") for i in self.style_list])
            )
        else:
            instruction = """
            Please describe the common writing styles of the two texts on the following linguistic levels:
            1. Morphological level
            2. Syntactic level
            3. Semantic level
            4. Discourse level
            """

        with DataDreamer(
            os.path.join(
                self.datadreamer_path,
                "cluster",
                cluster_method,
                self.model_name,
                "describe_document_cluster_styles",
            )
        ):
            results_dss = []
            for id, documents in zip(list_of_ids, list_of_documents):
                ds = DataSource(
                    "document_group",
                    Dataset.from_list(
                        [
                            {
                                "id": id,
                                "inputs": "\n\n".join(
                                    [
                                        "Text {}: {}".format(i + 1, d)
                                        for i, d in enumerate(x)
                                    ]
                                ),
                                "document_lists": x,
                            }
                            for x in documents
                        ]
                    ),
                )
                ds_desc = ProcessWithPrompt(
                    "Describe Documents Styles",
                    inputs={"inputs": ds.output["inputs"]},
                    args={
                        "llm": self.model,
                        "n": 1,
                        "max_new_tokens": self.max_new_tokens,
                        "instruction": instruction,
                    },
                    outputs={"generations": "generations"},
                ).select_columns(["generations"])

                ds_desc = ds_desc.map(
                    lambda row: {"style_list": process_style_desc(row["generations"])}
                )

                zipped_step = zipped(ds, ds_desc)
                results_dss.append(zipped_step)

            concat_step = concat(*results_dss, name="documents_style_description")

            results_ds = concat_step.output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            return pandas_df

    def describe_documents_writing_styles(self, list_of_ids, list_of_documents):
        """
        Describe the writing styles of a list of documents.

        Parameters:
            list_of_ids (list of str): List of document IDs.
            list_of_documents (list of str): List of document texts corresponding to the IDs.

        Returns:
            pandas.DataFrame: DataFrame containing the extracted writing style features.
        """
        with DataDreamer(
            os.path.join(
                self.datadreamer_path, self.model_name, "describe_document_styles"
            )
        ):
            datasets_list = []
            ds = DataSource(
                "documents",
                Dataset.from_list(
                    [
                        {"inputs": doc, "documentID": doc_id}
                        for doc_id, doc in zip(list_of_ids, list_of_documents)
                    ]
                ),
            )

            ds_desc = ProcessWithPrompt(
                "Describe Document Styles",
                inputs={"inputs": ds.output["inputs"]},
                args={
                    "llm": self.model,
                    "n": 1,
                    "max_new_tokens": self.max_new_tokens,
                    "instruction": self.prompts["describe_documents_writing_styles"][
                        "instruction"
                    ],
                },
                outputs={"generations": "generations"},
            ).select_columns(["generations"])

            ds_desc = ds_desc.map(
                lambda row: {"style_description": extract_feats(row["generations"])}
            )
            zipped_step = zipped(ds, ds_desc)
            datasets_list.append(zipped_step)

            concat_step = concat(*datasets_list, name="hrs_concat_style_descriptions")

            results_ds = concat_step.output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            return pandas_df

    def shorten_sentences(self, sentences, datadreamer_path=None):
        """
        Shorten sentences to be more generic and concise.

        Parameters:
            sentences (list of str): List of sentences to shorten.
            datadreamer_path (str, optional): Custom path for DataDreamer output. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing the shortened sentences.
        """
        instruction = """
        Please rewrite the provided sentence to be more generic and a short sentence that does not contain specifics or examples.
        """

        if datadreamer_path is None:
            save_dir = os.path.join(
                self.datadreamer_path, self.model_name, "shorten_sentences"
            )
        else:
            save_dir = os.path.join(
                self.datadreamer_path, self.model_name, datadreamer_path
            )

        with DataDreamer(save_dir):
            datasets_list = []
            ds = DataSource(
                "sentences",
                Dataset.from_list([{"inputs": "[SENTENCE]: " + x} for x in sentences]),
            )

            ds_desc = ProcessWithPrompt(
                "Shorten Sentences",
                inputs={"inputs": ds.output["inputs"]},
                args={
                    "llm": self.model,
                    "n": 1,
                    "max_new_tokens": self.max_new_tokens,
                    "instruction": instruction,
                },
                outputs={"generations": "generations"},
            ).select_columns(["generations"])

            zipped_step = zipped(ds, ds_desc)
            datasets_list.append(zipped_step)

            concat_step = concat(*datasets_list, name="hrs_concat_style_desc")

            results_ds = concat_step.output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            return pandas_df

    def summarize_sentences(self, list_of_sentences, instruction_key):
        """
        Summarize a list of sentences into a single paragraph.

        Parameters:
            list_of_sentences (list of list of str): List of sentences to summarize.

        Returns:
            pandas.DataFrame: DataFrame containing the summarized sentences.
        """
        with DataDreamer(
            os.path.join(
                self.datadreamer_path,
                "summarize",
                self.model_name,
                "summarize_sentences",
            )
        ):
            ds = DataSource(
                "sentences",
                Dataset.from_list([{"inputs": x} for x in list_of_sentences]),
            )
            ds_desc = ProcessWithPrompt(
                "Summarize Sentences",
                inputs={"inputs": ds.output["inputs"]},
                args={
                    "llm": self.model,
                    "n": 1,
                    "max_new_tokens": self.max_new_tokens,
                    "instruction": self.prompts["summarize_sentences"][instruction_key],
                },
                outputs={"generations": "generations"},
            ).select_columns(["generations"])

            zipped_step = zipped(ds, ds_desc)

            results_ds = zipped_step.output.dataset
            pandas_df = Dataset.from_generator(
                partial(gen_from_iterable_dataset, results_ds),
                features=results_ds.features,
            ).to_pandas()

            return pandas_df


#####################
# REFINING FEATURES #
#####################
def refine_features(
    valid_filtered_styles, mis, epsilon=10, split_threshold=0.5, merge_threshold=0.8
):
    """
    Refines features by clustering and merging similar styles based on word differences and affinity scores.

    Parameters:
    - valid_filtered_styles (list of str): List of styles to refine.
    - mis (object): Mutual Information Score object to compute similarity between styles.
    - epsilon (float, optional): Maximum distance between two samples for them to be considered as in the same neighborhood (default is 10).
    - split_threshold (float, optional): Threshold to decide whether to split a cluster (default is 0.5).
    - merge_threshold (float, optional): Threshold to decide whether to merge clusters (default is 0.8).

    Returns:
    - refined_styles (list of list of str): List of refined style groups.

    Steps:
    1. Construct Dissimilarity Matrix:
       - Compute pairwise word differences.
       - Store distances in a matrix.
    2. Clustering:
       - Apply DBSCAN clustering algorithm to the distance matrix.
       - Group sentences by cluster labels.
    3. Compute Mutual Information Scores:
       - For each cluster, compute MIS for pairs of styles.
    4. Split Clusters:
       - Split clusters if the average affinity is below split_threshold.
       - Keep track of removed indices.
    5. Merge Clusters:
       - Iteratively merge clusters if affinity scores exceed merge_threshold.
       - Continue merging until no more pairs can be merged.

    Example Usage:
    - Given a list of styles and a MIS object, refine the styles into coherent groups.

    """
    pair_to_distance = {}
    distance_matrix = np.zeros((len(valid_filtered_styles), len(valid_filtered_styles)))

    for index_1, sentence_1 in tqdm(
        enumerate(valid_filtered_styles),
        desc="Constructing Dissimilarity Matrix",
        total=len(valid_filtered_styles),
        ascii=True,
        leave=False,
    ):
        for index_2, sentence_2 in enumerate(valid_filtered_styles[index_1 + 1 :]):
            distance = word_difference(sentence_1, sentence_2)
            pair_to_distance[(sentence_1, sentence_2)] = distance
            distance_matrix[index_1, index_2 + index_1 + 1] = distance
            distance_matrix[index_2 + index_1 + 1, index_1] = distance

    clustering = DBSCAN(eps=epsilon, min_samples=2).fit(distance_matrix)

    sentence_dict = defaultdict(list)
    for key, sentence in zip(clustering.labels_, valid_filtered_styles):
        sentence_dict[key].append(sentence)
    sentence_dict = dict(sentence_dict)

    cluster_to_mis = {}
    sentence_dict = {x[0]: x[1] for x in sentence_dict.items() if len(x[1]) < 10}

    for cluster in tqdm(
        sentence_dict, total=len(sentence_dict), ascii=True, leave=False
    ):
        if cluster == -1:
            continue

        combinations = list(itertools.combinations(sentence_dict[cluster], 2))
        cluster_to_mis[cluster] = mis.compute(
            column(combinations, 0), column(combinations, 1), verbose=False
        )

    refined_styles = []

    for cluster in cluster_to_mis:
        refined_style = []
        removed_indices = []

        all_styles = sentence_dict[cluster]
        affinity_graph = np.zeros((len(all_styles), len(all_styles)))

        combinations = list(itertools.combinations(sentence_dict[cluster], 2))
        for index, (style_1, style_2) in enumerate(combinations):
            affinity_graph[all_styles.index(style_1), all_styles.index(style_2)] = (
                cluster_to_mis[cluster][index]
            )
            affinity_graph[all_styles.index(style_2), all_styles.index(style_1)] = (
                cluster_to_mis[cluster][index]
            )

        for index, row in enumerate(affinity_graph):
            if np.sum(row) / (len(row) - 1) <= split_threshold:
                refined_style.append([all_styles[index]])
                removed_indices.append(index)

        remaining_styles = [
            i for index, i in enumerate(all_styles) if index not in removed_indices
        ]
        if remaining_styles != []:
            refined_style.append(remaining_styles)

        refined_styles.extend(refined_style)

    num_iterations = 0

    while True:
        # Ideally, we would want to compute the affinity scores between all combinations of styles
        # Here, we approximate this by taking a style from the group as the representative
        refined_styles_single_combinations = list(
            itertools.combinations([i[0] for i in refined_styles], 2)
        )
        affinity_score_single_combinations = mis.compute(
            column(refined_styles_single_combinations, 0),
            column(refined_styles_single_combinations, 1),
            verbose=False,
        )

        merge_graph = np.zeros((len(refined_styles), len(refined_styles)))

        for index, (index_1, index_2) in enumerate(
            list(itertools.combinations([i for i in range(len(refined_styles))], 2))
        ):
            if affinity_score_single_combinations[index] >= merge_threshold:
                merge_graph[index_1, index_2] = 1

        # Loop and merge
        merge_pairs = []
        already_merged = []
        for index_1, row in tqdm(
            enumerate(merge_graph),
            total=len(merge_graph),
            ascii=True,
            desc="Constructing Merge Pairs",
            leave=False,
        ):
            if index_1 in already_merged:
                continue

            row = row[index_1 + 1 :]
            for index_2, col in enumerate(row):
                if col == 1:
                    merge_pairs.append([index_1, index_1 + index_2 + 1])
                    merge_graph[:, index_1 + index_2 + 1] = 0
                    already_merged.append(index_1 + index_2 + 1)
                    break

        # If there are no pairs to merge, break
        print(f"# Iteration {num_iterations+1}: {len(merge_pairs)} Pairs to Merge.")

        if merge_pairs == []:
            break

        refined_styles = merge_sublists(refined_styles, merge_pairs)
        num_iterations += 1

        sys.stdout.write("\033[F\033[K")
        sys.stdout.flush()

    return refined_styles


def process_style_features(
    style_generator, df, datadreamer_path, column_name, output_clm, output_path
):
    # Shorten features
    attributes_set = df[column_name].unique()

    res = style_generator.shorten_sentences(attributes_set, datadreamer_path)
    # Make sure that all shortend features are valid sentences
    valid_shortend_sentences = {
        x: is_sentence_finished(x) for x in tqdm(set(res.generations.tolist()))
    }

    res["valid_sentence"] = res.generations.apply(lambda x: valid_shortend_sentences[x])
    feature_to_shortend = {
        row["inputs"].replace("[SENTENCE]: ", ""): row["generations"]
        for _, row in res[res.valid_sentence].iterrows()
    }

    df[output_clm] = df[column_name].apply(
        lambda c: feature_to_shortend[c] if c in feature_to_shortend else c
    )
    df.to_csv(output_path, index=False)

    return df