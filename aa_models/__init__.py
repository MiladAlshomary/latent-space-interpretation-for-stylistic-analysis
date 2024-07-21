import os
import sys

import numpy as np
import pandas as pd


def get_model(model, device=None):
    current_dir = os.path.dirname(__file__)

    # Add the directory to sys.path
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    if model == "aa_model-luar":
        from luar_pausit import LUAR_PAUSIT

        return LUAR_PAUSIT(device)

    elif model == "luar-mud":
        from luar_mud import LUAR_MUD

        return LUAR_MUD()

    elif model == "luar-crud":
        from luar_crud import LUAR_CRUD

        return LUAR_CRUD()

    elif model == "wegmann":
        from wegmann import WEGMANN

        return WEGMANN()


##################
# EMBEDDING TEXT #
##################
def get_proj_matrices(model, mode, style_documents_path):
    """
    Inputs:

    Returns:
        proj_dims: dictionary with indices to styles mapping
        proj_matrix:
    """
    style_documents = pd.read_pickle(style_documents_path)

    if mode == "generate":
        # `style_documents`: column 1 is style, column 2 is list of generated documents for each style
        proj_dims = {
            index: x for index, x in enumerate(style_documents["t_style"].tolist())
        }

        proj_matrix = []
        for paras in style_documents["paragraphs"].tolist():
            embeddings = model.encode(list(paras))
            style_embedding = np.mean(embeddings, axis=0)

            proj_matrix.append(style_embedding)

    elif mode == "rewrite":
        styles = set(style_documents["t_style"])
        proj_dims = {index: x for index, x in enumerate(styles)}

        proj_matrix = []
        for style in styles:
            # Fetch list of original and rewritten paragraphs
            corresponding_docs = style_documents[style_documents["t_style"] == style]

            original = list(corresponding_docs["text"])
            rewritten = list(corresponding_docs["paragraphs"])

            # Encode all paragraphs corresponding to style
            original_embeddings = model.encode(original)
            rewritten_embeddings = model.encode(rewritten)

            style_embedding = np.mean(
                rewritten_embeddings - original_embeddings, axis=0
            )

            proj_matrix.append(style_embedding)

    proj_matrix = np.array(proj_matrix)

    return proj_matrix, proj_dims


def interp_docs(model, proj_matrix, documents):
    """ """
    documents_embeddings = model.encode(documents)
    doc_interp = [proj_matrix.dot(e) for e in documents_embeddings]

    return documents_embeddings, doc_interp
