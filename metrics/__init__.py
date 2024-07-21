import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import average_precision_score, roc_curve, ndcg_score


##################################
# Authorship Attribution Metrics #
##################################
def compute_eer(label, pred):
    """ """
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, _ = roc_curve(label, pred)
    fnr = 1 - tpr

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2

    return eer


def compute_model_performance(embed_model, df, proj_matrices):
    """ """
    # Embed documents with AA model
    documents_embeddings = embed_model.encode(df.fullText.tolist())

    # Project document embeddings onto interpretable bases
    interp_embeddings = []
    for proj_matrix in proj_matrices:
        interp_embeddings.append([proj_matrix.dot(e) for e in documents_embeddings])

    # Compute performance metrics
    interp_documents_pairwise_sims = np.mean(
        np.array([cosine_similarity(x, x) for x in interp_embeddings]), axis=0
    )
    index_to_author_id = {i: x for i, x in enumerate(df.authorID.tolist())}

    prec_interp = []
    eer_interp = []
    ndcg_interp = []

    for index, row in df.iterrows():
        author_id = row["authorID"]

        y_true = [
            1 if author_id == a_id else 0
            for i, a_id in index_to_author_id.items()
            if i != index
        ]
        y_interp_score = [
            interp_documents_pairwise_sims[index][i]
            for i, a_id in index_to_author_id.items()
            if i != index
        ]

        prec_interp.append(average_precision_score(y_true, y_interp_score))
        try:
            eer_interp.append(compute_eer(y_true, y_interp_score))
        except ValueError:
            eer_interp.append(1)

        ndcg_interp.append(ndcg_score([y_true], [y_interp_score]))

    return (
        round(np.mean(eer_interp), 3),
        round(np.mean(prec_interp), 3),
        round(np.mean(ndcg_interp), 3),
    )