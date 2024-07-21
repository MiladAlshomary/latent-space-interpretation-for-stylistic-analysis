import os
import re
import ast
import random
from collections import Counter

import spacy
import Levenshtein
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from tqdm import tqdm
from tabulate import tabulate
from nltk.corpus import wordnet as wn
from plotly.subplots import make_subplots
from spacy.matcher import Matcher


# Load the spaCy model
nlp = None


####################
# HELPER FUNCTIONS #
####################
class InvalidModelNameError(Exception):
    """
    Exception raised for errors in the model name format.

    Attributes:
        message (str): Explanation of the error. Defaults to "Model name does not follow the format '[FAMILY]:[MODEL NAME].'"
    """

    def __init__(
        self, message="Model name does not follow the format '[FAMILY]:[MODEL NAME].'"
    ):
        """
        Initialize the InvalidModelNameError with an optional error message.

        Parameters:
            message (str): Explanation of the error.
        """
        self.message = message
        super().__init__(self.message)

def safe_parse(doc_id):
    if isinstance(doc_id, list):
        return doc_id
    try:
        return ast.literal_eval(doc_id)
    except (ValueError, SyntaxError):
        return doc_id.strip("[]").replace("'", "").split(", ")


def gen_from_iterable_dataset(iterable_ds):
    yield from iterable_ds


def column(matrix, i):
    return [row[i] for row in matrix]


def partition(obj, num_partitions):
    """ """
    chunks = int(len(obj) // num_partitions)

    chunk_size = 0
    chunk_list = []
    buf = []
    for i in obj:
        if chunk_size >= chunks:
            chunk_list.append(buf)
            chunk_size = 0
            buf = []

        buf.append(i)
        chunk_size += 1

    if len(buf) != 0:
        chunk_list.append(buf)

    return chunk_list


def replace_first_word(input_string, new_word):
    # Extract the last portion after the last backslash
    last_part = os.path.basename(input_string)

    # Split by "-" and replace the first word
    parts = last_part.split("-")
    parts[0] = new_word

    # Join the parts back together
    new_last_part = "-".join(parts)

    # Replace the last portion in the original string with the new last portion
    new_string = os.path.join(os.path.dirname(input_string), new_last_part)

    return new_string


def check_model_name_format(s):
    """
    Check if the given string follows the format '[FAMILY]:[MODEL NAME]'.

    Parameters:
        s (str): The string to be checked.

    Raises:
        InvalidModelNameError: If the string does not follow the format '[FAMILY]:[MODEL NAME]'.

    Examples:
        >>> check_model_name_format("ABC:123")
        (No exception raised)

        >>> check_model_name_format("ABC123")
        Traceback (most recent call last):
            ...
        InvalidModelNameError: Model name does not follow the format '[FAMILY]:[MODEL NAME].'
    """
    pattern = r"^[^:]+:[^:]+$"
    if not re.match(pattern, s):
        raise InvalidModelNameError()


def extract_feats(desc):
    """
    Extract writing style features from the given description.

    Parameters:
        desc (str): Description text containing writing style attributes.

    Returns:
        dict: Dictionary containing extracted features for each writing style level.
    """
    output = {
        "Morphological Level": ["other"],
        "Syntactic Level": ["other"],
        "Semantic Level": ["other"],
        "Discourse Level": ["other"],
    }

    paras = desc.split("\n\n")

    if paras and "Here are the writing style attributes of the given text" in paras[0]:
        paras = paras[1:]

    if not paras:
        return output

    for i in range(0, len(paras), 2):
        if i + 1 >= len(paras):
            continue

        category = re.sub(r"\*\*", "", paras[i]).strip().replace(":", "")
        features = [
            f.replace("-", "").replace("*", "").strip()
            for f in paras[i + 1].split("\n")
        ]

        if category in output:
            output[category] = features
        else:
            print(f"{category} not found in the keys")

    return output


def find_first_minimal_change(data, threshold):
    """
    Find the first key where the change in all three values compared to the previous key is minimal.

    Parameters:
    data (dict): A dictionary where keys are integers and values are tuples of three floats.
    threshold (float): Maximum allowed change for each of the three values.

    Returns:
    int: The first key where the change in all three values is minimal (less than the threshold).
         Returns None if no such key is found.

    Example:
    --------
    >>> data = {
    ...     1: (0.512, 0.001, 0.145),
    ...     3: (0.299, 0.01, 0.172),
    ...     8: (0.246, 0.021, 0.196),
    ...     21: (0.126, 0.126, 0.33),
    ...     ...
    ... }
    >>> find_first_minimal_change(data)
    21

    Notes:
    ------
    - The dictionary should have integer keys and tuple values with exactly three float elements.
    - The function assumes that the dictionary keys are sortable (e.g., integers).
    - The function will only compare values if there is a previous value to compare with.
    - If no key meets the minimal change condition, the function returns None.
    """
    previous_values = None

    for key in sorted(data.keys(), reverse=True):
        current_values = data[key]

        if previous_values is not None:
            changes = [
                abs(current - previous)
                for current, previous in zip(current_values, previous_values)
            ]
            if all(change < threshold for change in changes):
                return key

        previous_values = current_values

    return None


############
# PLOTTING #
############
def generate_colors(n):
    # Use the 'Viridis' color scale for a diverse and aesthetically pleasing set of colors
    colors = pc.sample_colorscale(pc.sequential.Viridis, np.linspace(0, 1, n))
    return colors


def plot_dimensionality_reduction(
    method, embeddings, cluster_labels, unique_labels, color_map
):
    fig = go.Figure()

    for label in unique_labels:
        idx = [i for i, l in enumerate(cluster_labels) if l == label]
        fig.add_trace(
            go.Scatter(
                x=embeddings[idx, 0],
                y=embeddings[idx, 1],
                mode="markers",
                marker=dict(size=5, color=color_map[label], opacity=1),
            )
        )

    fig.update_layout(
        title=f"{method}",
        title_x=0.5,
        title_font_size=30,
        margin=dict(l=10, r=10, b=0, t=50),
        width=600,
        height=600,
        showlegend=False,
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        nticks=20,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        nticks=20,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickfont=dict(size=10),
    )

    fig.show()


def plot_aa_performance(results, combined=False):
    """ """
    # Filter bad methods
    results = {key: value for (key, value) in results.items() if value != {}}
    methods = list(results.keys())

    y_axes_label = ["EER", "NDCG", "AP"]
    label_to_axes_range = {"EER": [0, 0.5], "NDCG": [0, 0.3], "AP": [0, 0.5]}
    best_results = {i: {j: None for j in y_axes_label} for i in methods}
    best_results_assignments = {i: None for i in methods}

    if not combined:
        fig = make_subplots(
            rows=3,
            cols=len(methods),
            subplot_titles=methods,
            vertical_spacing=0.05,
            horizontal_spacing=0.01,
        )
    else:
        color_map = {
            label: generate_colors(len(methods))[index]
            for index, label in enumerate(methods)
        }

        fig = make_subplots(
            rows=3, cols=1, subplot_titles=y_axes_label, vertical_spacing=0.05
        )

    for method_index, method in enumerate(results):
        sorted_results = {k: results[method][k] for k in sorted(results[method].keys())}

        x_axis = list(sorted_results.keys())
        y_axes = list(sorted_results.values())

        optimal_points = []
        for index, label in enumerate(y_axes_label):
            optimal_points.append(
                find_optimal_stable_point(
                    x_axis, column(y_axes, index), best=(label != "EER")
                )
            )
        best_point = round(np.mean(optimal_points))
        next_best_point = find_next_largest_or_equal(x_axis, best_point)
        best_results[method]["Clusters"] = next_best_point
        best_results_assignments

        for index, label in enumerate(y_axes_label):
            y_axis = column(y_axes, index)
            best_results[method][label] = y_axis[x_axis.index(next_best_point)]

            if not combined:
                fig.add_shape(
                    type="line",
                    x0=next_best_point,
                    x1=next_best_point,
                    y0=0,
                    y1=label_to_axes_range[label][-1],
                    line=dict(color="Red", width=2),
                    row=index + 1,
                    col=method_index + 1,
                )

                fig.add_shape(
                    type="line",
                    x0=optimal_points[index],
                    x1=optimal_points[index],
                    y0=0,
                    y1=label_to_axes_range[label][-1],
                    line=dict(color="Gray", width=2),
                    row=index + 1,
                    col=method_index + 1,
                )

                fig.add_trace(
                    go.Scatter(x=x_axis, y=y_axis, mode="lines+markers", name=label),
                    row=index + 1,
                    col=method_index + 1,
                )

                fig.update_yaxes(
                    range=label_to_axes_range[label],
                    row=index + 1,
                    col=method_index + 1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_axis,
                        y=y_axis,
                        mode="lines+markers",
                        name=method,
                        showlegend=(index == 0),
                        marker=dict(size=5, color=color_map[method], opacity=1),
                    ),
                    row=index + 1,
                    col=1,
                )

                fig.update_yaxes(range=label_to_axes_range[label], row=index + 1, col=1)

        if not combined:
            # Add annotations for each row title
            for i, title in enumerate(y_axes_label):
                fig.add_annotation(
                    dict(
                        text=title,
                        x=-0.02,
                        y=1 - (i / 3 + 1 / 6),
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=20),
                    )
                )

    fig.update_layout(
        title="Authorship Attribution Performance",
        title_x=0.5,
        title_font_size=30,
        margin=dict(l=130, r=10, b=0, t=80),
        showlegend=combined,
        width=300 * len(methods) if not combined else 900,
        height=600 if not combined else 900,
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        nticks=20 if not combined else 10,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        nticks=20,
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        tickfont=dict(size=10),
    )

    fig.show()

    return fig, best_results


def plot_dists(
    X_embedded, z_params, marker_sizes=8, show=False, show_tqdm=False, save=False
):
    """
    Create 3D histogram-like plots from embedded data and corresponding z values.

    Parameters:
    - X_embedded: np.ndarray
        2D array of shape (n_samples, 2) representing x and y coordinates.
    - z_params: dict
        Dictionary containing the following keys:
        - z_values: np.ndarray
            2D array of shape (n_samples, n_features) representing z values for each point.
        - z_values_labels: list of str
            List of legend names corresponding to each feature in z_values.
        - z_value_name: str
            Name used for saving the output files.
        - z_max: float, optional
            Maximum value for the z-axis range.
        - z_values_per_plot: int, optional
            Number of z values to plot per figure. Use "all" to plot all in one figure.
    - marker_sizes: int, optional, default=8
        Size of the markers used in the plot.
    - show: bool, optional, default=False
        Whether to show plots interactively.
    - show_tqdm: bool, optional, default=False
        Whether to show a progress bar for plotting.
    - save: bool, optional, default=False
        Whether to save the plots as HTML files.

    Returns:
    - None
    """

    def prepare_bars_data(x_points, y_points, z_values):
        x_data, y_data, z_data = [], [], []
        for x, y, z in zip(x_points, y_points, z_values):
            x_data.extend([x, x, None])
            y_data.extend([y, y, None])
            z_data.extend([0, z, None])
        return x_data, y_data, z_data

    def create_3d_plot(X_embedded, z_values_subset, legend_names, z_max):
        fig = go.Figure()
        bar = (
            tqdm(
                range(z_values_subset.shape[1]),
                ascii=True,
                total=z_values_subset.shape[1],
            )
            if show_tqdm
            else range(z_values_subset.shape[1])
        )

        for i in bar:
            z_col = z_values_subset[:, i]
            x_data, y_data, z_data = prepare_bars_data(
                X_embedded[:, 0], X_embedded[:, 1], z_col
            )
            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode="lines",
                    line=dict(width=marker_sizes * 2),
                    name=legend_names[i],
                    visible="legendonly",
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=X_embedded[:, 0],
                y=X_embedded[:, 1],
                z=[0] * len(X_embedded),
                mode="markers",
                marker=dict(size=marker_sizes // 2, color="gray"),
                name="Base Points",
                visible=True,
                showlegend=False,
            )
        )

        fig.update_layout(
            scene=dict(zaxis=dict(title="Style Strength", range=[0, z_max])),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(y=0.5, yanchor="middle"),
        )

        return fig

    z_values = z_params["z_values"]
    z_values_labels = z_params["z_values_labels"]
    z_value_name = z_params["z_value_name"]
    z_max = z_params.get("z_max", 1)
    z_values_per_plot = z_params.get("z_values_per_plot", 10)

    if z_values_per_plot == "all":
        fig = create_3d_plot(X_embedded, z_values, z_values_labels, z_max)

        if show:
            fig.show()
        if save:
            fig.write_html(f"rep={z_value_name}.html")
    else:
        num_plots = z_values.shape[1] // z_values_per_plot
        batches = range(num_plots)

        for i in batches:
            start_idx, end_idx = i * z_values_per_plot, (i + 1) * z_values_per_plot
            z_values_subset = z_values[:, start_idx:end_idx]
            fig = create_3d_plot(
                X_embedded, z_values_subset, z_values_labels[start_idx:end_idx], z_max
            )

            if show:
                fig.show()
            if save:
                fig.write_html(f"rep={z_value_name}-batch_{i}.html")


#############################
# NOTEBOOK HELPER FUNCTIONS #
#############################
def find_all_extrema(y, best):
    """
    Find all local minima and maxima in the y values.

    Args:
    y (list): y-axis values.

    Returns:
    list: Indices of local minima and maxima.
    """
    if best:
        best_value = np.max(y)
    else:
        best_value = np.min(y)

    extrema_indices = [y.index(i) for i in y if i == best_value]

    return extrema_indices


def find_stable_point(x, y, threshold, window):
    """
    Find the point after which the values don't change much.

    Args:
    x (list): x-axis values.
    y (list): y-axis values.
    threshold (float): The change threshold to consider a value as stable.
    window (int): Number of consecutive points to consider for stability.

    Returns:
    int: The x value after which the changes are consistently small.
    """
    differences = np.abs(np.diff(y))
    for i in range(len(differences) - window + 1):
        if np.all(differences[i : i + window] < threshold):

            return x[i + window]
    return x[-1]  # If no stable point is found, return the last x value


def find_optimal_stable_point(x, y, best):
    """
    Find the optimal stable point where the score is highest (or lowest) and changes minimally,
    or find the point after which the changes are consistently small.

    Args:
    x (list): x-axis values.
    y (list): y-axis values.

    Returns:
    int: The x value representing the optimal stable point.
    """
    # Find all local minima and maxima
    extrema_indices = find_all_extrema(y, best)

    if not extrema_indices:
        # If no extrema, directly find the stable point
        return find_stable_point(x, y, np.std(np.diff(y)), 3)

    # Find the extrema with the lowest value
    lowest_extrema_index = extrema_indices[np.argmin([y[i] for i in extrema_indices])]

    if lowest_extrema_index == len(x) - 1:
        # If the lowest extrema is at the end, find the stable point
        return find_stable_point(x, y, np.std(np.diff(y)), 3)

    return x[lowest_extrema_index]


def find_next_largest_or_equal(nums, target):
    """
    Find the value in nums that is the closest to target and is equal to or strictly larger.

    Args:
    nums (list): List of integers.
    target (int): The target integer.

    Returns:
    int: The closest value that is equal to or strictly larger than the target.
    """
    next_largest = None

    for num in nums:
        if num >= target:
            if next_largest is None or num < next_largest:
                next_largest = num

    return next_largest


def merge_sublists(sublists, pairs):
    """
    Merges sublists based on provided pairs of indices.

    Parameters:
    - sublists (list of list of str): The original list of sublists to merge.
    - pairs (list of tuple of int): List of index pairs indicating which sublists to merge.

    Returns:
    - list of list of str: The merged list of sublists.

    Steps:
    1. Create a copy of the original sublists to avoid modifying the input.
    2. Initialize a set to keep track of indices that have been merged.
    3. Iterate over each pair of indices.
       - If either index in the pair has already been merged, skip to the next pair.
       - Merge the sublist at the first index with the sublist at the second index.
       - Mark the second index as merged.
    4. Create the final list of sublists, excluding the merged indices.
    5. Return the final list of sublists.

    Example Usage:
    - Given sublists = [['a', 'b'], ['c', 'd'], ['e']] and pairs = [(0, 1)], the function will return [['a', 'b', 'c', 'd'], ['e']].

    """
    merged_sublists = sublists[:]
    merged_indices = set()

    for i, j in pairs:
        if i in merged_indices or j in merged_indices:
            continue

        merged_sublists[i] = merged_sublists[i] + merged_sublists[j]
        merged_indices.add(j)

    final_sublists = [
        sublist
        for index, sublist in enumerate(merged_sublists)
        if index not in merged_indices
    ]

    return final_sublists


def remove_infrequent_items(input_list, min_occurrences=3):
    """
    Removes items from the input list that appear less than a specified number of times.

    Parameters:
    - input_list (list of str): The original list of strings.
    - min_occurrences (int, optional): The minimum number of occurrences required for an item to be kept (default is 3).

    Returns:
    - list of str: The filtered list with infrequent items removed.
    """
    # Count occurrences of each item in the list
    item_counts = Counter(input_list)

    # Filter out items that appear less than min_occurrences times
    filtered_list = [
        item for item in input_list if item_counts[item] >= min_occurrences
    ]

    return filtered_list


def print_clusters(clusters, num_samples=5):
    for index, cluster in enumerate(clusters):
        sampled_keys = random.sample(cluster, min(num_samples, len(cluster)))

        print(
            tabulate(
                [[" - " + i] for i in sampled_keys],
                headers=[f"Samples for Cluster {index+1}:"],
                tablefmt="simple",
            ),
            end="\n\n",
        )


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Check if the nested dictionary contains keys that are not strings
            if all(isinstance(key, str) for key in v.keys()):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)


def get_np(text):
    output = []
    global nlp
    if nlp == None:
        nlp = spacy.load("en_core_web_sm")

    matcher = Matcher(nlp.vocab)
    patterns = [
        [{"POS": "ADJ"}, {"POS": "NOUN", "OP": "+"}, {"POS": "ADV", "OP": "?"}],
        [{"POS": "NOUN"}, {"TEXT": "is"}, {"POS": "ADV", "OP": "?"}, {"POS": "ADJ"}],
        [{"POS": "NOUN"}, {"TEXT": "is"}, {"POS": "ADV", "OP": "?"}, {"POS": "NOUN"}],
    ]
    matcher.add("demo", patterns)
    doc = nlp(text)
    matches = matcher(doc)
    for _, start, end in matches:
        span = doc[start:end]
        output.append(span.text.lower())

    # Filter output
    final_output = []
    for item in sorted(output, key=lambda x: -len(x)):
        if any([item in x for x in final_output]):
            continue
        final_output.append(item)

    return final_output


####################
# DISTANCE METRICS #
####################
def word_based_levenshtein(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Join the word lists into strings with words separated by space
    string1 = " ".join(words1)
    string2 = " ".join(words2)

    return Levenshtein.distance(string1, string2)


def word_difference(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()

    # Pad the shorter list with empty strings to make the lengths equal
    max_len = max(len(words1), len(words2))
    words1.extend([""] * (max_len - len(words1)))
    words2.extend([""] * (max_len - len(words2)))

    # Count the number of differing words
    difference_count = sum(1 for w1, w2 in zip(words1, words2) if w1 != w2)

    return difference_count


####################
# MANUAL FILTERING #
####################


def is_sentence_finished(sentence):
    def is_valid_word(word):
        return bool(wn.synsets(word))

    global nlp
    if nlp == None:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(sentence)

    # Check if the last token is a proper punctuation mark
    if sentence.strip()[-1] in ".!?":
        return True

    # Check for typical unfinished sentence patterns
    if len(doc) > 0:
        last_token = doc[-1]
        if last_token.pos_ in {
            "CCONJ",
            "SCONJ",
            "ADP",
            "PART",
            "DET",
            "PRON",
            "AUX",
            "INTJ",
            "ADJ",
            "ADV",
            "NUM",
        }:
            return False
        if last_token.dep_ in {
            "prep",
            "mark",
            "aux",
            "det",
            "poss",
            "amod",
            "advmod",
            "nummod",
        }:
            return False

    # Common unfinished phrases
    unfinished_phrases = [
        "such as",
        "including",
        "like",
        "especially",
        "without",
        "e.g.",
        "i.e.",
        "among others",
        "such that",
        "such",
        "while",
        "where",
        "when",
        "because",
        "as",
        "even though",
        "for example",
        "such as",
    ]
    for phrase in unfinished_phrases:
        if sentence.strip().lower().endswith(phrase):
            return False

    # Check for truncated words
    if len(doc) > 1 and len(doc[-1].text) <= 2 and not doc[-1].is_punct:
        return False

    # Check if the last word is valid using WordNet
    last_word = sentence.strip().split()[-1].strip(".!?")
    if not is_valid_word(last_word.lower()):
        return False

    # Unmatched parentheses, quotes, or brackets
    if sentence.count("(") != sentence.count(")"):
        return False
    if sentence.count('"') % 2 != 0:
        return False
    if sentence.count("'") % 2 != 0:
        return False
    if sentence.count("[") != sentence.count("]"):
        return False
    if sentence.count("{") != sentence.count("}"):
        return False

    # Sentence ending with an incomplete clause (e.g., "with", "for", "to", etc.)
    incomplete_endings = {
        "with",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "about",
        "over",
        "under",
        "after",
        "between",
    }
    if len(doc) > 0 and doc[-1].text.lower() in incomplete_endings:
        return False

    # Sentence ending with certain adverbs or adjectives suggesting continuation
    unfinished_adverbs_adjectives = {
        "simply",
        "merely",
        "only",
        "almost",
        "nearly",
        "barely",
        "seldom",
        "informal",
        "directly",
        "regularly",
    }
    if len(doc) > 0 and doc[-1].text.lower() in unfinished_adverbs_adjectives:
        return False

    # Sentence ending with possessives (e.g., "author's", "student's")
    if len(doc) > 0 and doc[-1].tag_ in {"POS", "PRP$", "WP$"}:
        return False

    # Sentence ending with verbs suggesting continuation
    unfinished_verbs = {
        "using",
        "employing",
        "considering",
        "following",
        "including",
        "utilizing",
        "featuring",
    }
    if len(doc) > 0 and doc[-1].text.lower() in unfinished_verbs:
        return False

    # Sentence ending with common introductory words (e.g., "because", "although")
    introductory_words = {"because", "although", "since", "unless", "whereas"}
    if len(doc) > 0 and doc[-1].text.lower() in introductory_words:
        return False

    # Sentence ending with interjections or discourse markers
    discourse_markers = {"well", "so", "like", "you know", "I mean"}
    if len(doc) > 0 and doc[-1].text.lower() in discourse_markers:
        return False

    # Sentence ending with ellipsis
    if sentence.strip().endswith("..."):
        return False

    # Sentence ending with common abbreviations that suggest more context (e.g., "etc.")
    common_abbreviations = {"etc.", "viz.", "i.e.", "e.g."}
    if len(doc) > 0 and doc[-1].text.lower() in common_abbreviations:
        return False

    # Sentence with trailing conjunctions
    trailing_conjunctions = {"and", "or", "but"}
    if len(doc) > 1 and doc[-2].text.lower() in trailing_conjunctions:
        return False

    # Sentence with common openers without a following clause
    common_openers = {
        "according to",
        "based on",
        "due to",
        "in light of",
        "in accordance with",
        "as a result of",
    }
    for opener in common_openers:
        if (
            sentence.strip().lower().startswith(opener)
            and len(doc) <= len(opener.split()) + 1
        ):
            return False

    # If none of the above checks indicate an unfinished sentence, assume it is finished
    return True


def filter_verifier_response(verifier_response):
    """
    Determines the verdict of a response by analyzing the provided text.
    """
    good = []
    bad = []
    missing = []
    correct = []

    patterns = [
        (r"(paragraph 1 better|\"paragraph 1\" better|\[paragraph 1\] better)", True),
        (r"(paragraph 2 better|\"paragraph 2\" better|\[paragraph 2\] better)", False),
        (r"^(paragraph 1|the first paragraph)", True),
        (r"^(paragraph 2|the second paragraph)", False),
    ]

    for response in verifier_response:
        response = response.lower()
        found = False

        for pattern, is_paragraph_1 in patterns:
            if re.search(pattern, response):
                correct.append(is_paragraph_1)
                good.append(response)
                found = True
                break

        if not found:
            if "paragraph 1" in response or "paragraph 2" in response:
                correct.append("paragraph 1" in response)
                good.append(response)
            else:
                correct.append(False)
                missing.append(response)

    return correct, (good, bad, missing)
