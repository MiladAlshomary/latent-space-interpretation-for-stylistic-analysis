import os

import numpy as np
from absl import logging
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity


class Similarity:

    def __init__(
        self,
        query_features,
        candidate_features,
        query_labels,
        candidate_labels,
        input_dir,
    ):
        self.query_features = query_features
        self.candidate_features = candidate_features
        self.query_labels = query_labels
        self.candidate_labels = candidate_labels
        self.dataset_path = get_dataset_path(input_dir)

    def compute_similarities(self):
        logging.info("Computing cosine similarities")
        self.psimilarities = cosine_similarity(
            self.query_features, self.candidate_features
        )

    def save_ta2_output(self, output_dir, run_id, ta1_approach):
        logging.info("Saving similarities and labels")
        HST = os.path.basename(self.dataset_path).split("_TA2")[0]

        np.save(
            os.path.join(
                output_dir, HST + f"_TA2_query_candidate_attribution_scores_{run_id}"
            ),
            self.psimilarities,
        )

        fout = open(
            os.path.join(
                output_dir,
                HST + f"_TA2_query_candidate_attribution_query_labels_{run_id}.txt",
            ),
            "w+",
        )
        if self.query_labels[0][0] == "(":
            tuple_str = True
        else:
            tuple_str = False
        if not tuple_str:
            for label in self.query_labels:
                fout.write("('" + str(label) + "',)")
                fout.write("\n")
            fout.close()
        else:
            for label in self.query_labels:
                fout.write(label)
                fout.write("\n")
            fout.close()

        fout = open(
            os.path.join(
                output_dir,
                HST + f"_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt",
            ),
            "w+",
        )

        if not tuple_str:
            for label in self.candidate_labels:
                fout.write("('" + str(label) + "',)")
                fout.write("\n")
            fout.close()
        else:
            for label in self.candidate_labels:
                fout.write(label)
                fout.write("\n")
            fout.close()


def get_dataset_path(input_path):
    dataset_path = glob(os.path.join(input_path, "*queries*"))[0]
    return dataset_path
