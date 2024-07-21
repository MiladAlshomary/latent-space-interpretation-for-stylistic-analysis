import os

import numpy as np
from absl import logging
from glob import glob
import statistics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class KNN:

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

    def compute_knn_preds(self):
        logging.info("Fitting KNN")
        logging.info(f"Queries[0] shape: {len(self.query_features[0])}")
        model = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        model.fit(self.query_features, self.query_labels)
        predictions = model.predict(self.candidate_features)
        probs = model.predict_proba(self.candidate_features)

        # Log the prediction probabilities for the first few samples
        for i in range(
            min(5, len(probs))
        ):  # Log the first 5 samples or less if there are fewer
            logging.info(f"Sample {i+1} - Class Probabilities:\n{probs[i]}")

        # Init appropriate matrix
        num_documents = len(self.candidate_features)
        author_indices = {
            author: index for index, author in enumerate(self.query_labels)
        }
        self.author_matrix_preds = np.zeros((len(self.query_labels), num_documents))

        # Iterate over documents and fill in the matrix directly with swapped dimensions
        for i in range(num_documents):
            predicted_author = predictions[i]
            author_index = author_indices[predicted_author]
            self.author_matrix_preds[author_index, i] = 1

        accuracy = metrics.accuracy_score(self.candidate_labels, predictions)
        logging.info(f"Accuracy: {accuracy}")

    def save_ta2_output(self, output_dir, run_id, ta1_approach):
        logging.info("Saving similarities and labels")
        HST = os.path.basename(self.dataset_path).split("_")[0]

        np.save(
            os.path.join(
                output_dir, HST + f"_TA2_query_candidate_attribution_scores_{run_id}"
            ),
            self.author_matrix_preds,
        )

        fout = open(
            os.path.join(
                output_dir,
                HST + f"_TA2_query_candidate_attribution_query_labels_{run_id}.txt",
            ),
            "w+",
        )
        for label in self.query_labels:
            fout.write("('" + str(label) + "',)")
            fout.write("\n")
        fout.close()

        fout = open(
            os.path.join(
                output_dir,
                HST + f"_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt",
            ),
            "w+",
        )
        for label in self.candidate_labels:
            fout.write("('" + str(label) + "',)")
            fout.write("\n")
        fout.close()


def get_dataset_path(input_path):
    dataset_path = glob(os.path.join(input_path, "*TA2_queries*"))[0]
    return dataset_path
