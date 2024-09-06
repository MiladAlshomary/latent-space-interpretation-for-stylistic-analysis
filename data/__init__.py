import pandas as pd
import numpy as np

def get_aa_data(data_path, group_by_author=False):
    """
    Load data from a JSON file. Extract the first author ID.
    Group by author ID. Aggregate full texts and document IDs.
    Return a DataFrame.

    Parameters:
    data_path (str): Path to the JSON file.

    Returns:
    pd.DataFrame: DataFrame with aggregated data.

    Example:
    --------
    >>> data_path = "data.json"
    >>> df = get_aa_data(data_path)
    >>> print(df.head())
       authorID                                fullText       documentID
    0     12345  [Text of document 1, Text of document 2]  [ID1, ID2]
    1     67890  [Text of document 3, Text of document 4]  [ID3, ID4]

    Notes:
    ------
    - The JSON file should be in a line-delimited format.
    - Each record must contain 'authorIDs', 'fullText', and 'documentID'.
    - 'authorIDs' should be a list of IDs.
    """
    data_df = pd.read_json(data_path)
    if "authorID" not in data_df.columns:
        data_df["authorID"] = data_df["authorIDs"].str[0]
    
    if group_by_author:
        data_df = (
            data_df.groupby("authorID")
            .agg({"fullText": list, "documentID": list})
            .reset_index()
        )

    return data_df

def get_aa_data_from_original_format(data_path, groundtruth_path):
    def q_c_mapping(c_author):
        c_author_idx = candidate_authors.index(c_author)
        found_assignment = np.where(ground_truth_assignment[:,candidate_authors.index(c_author)] == 1)
        if len(found_assignment[0]) > 0:
            q_author_idx = found_assignment[0][0]    
            return query_authors[q_author_idx]
        else:
            return c_author
            
    queries_df = pd.read_json(data_path + '_queries.jsonl', lines=True)
    candidates_df = pd.read_json(data_path + '_candidates.jsonl', lines=True)
    queries_df['authorID']  = queries_df['authorIDs'].apply(lambda x : x[0])
    candidates_df['authorSetID']  = candidates_df['authorSetIDs'].apply(lambda x : x[0])
    
    ground_truth_assignment = np.load(open(groundtruth_path + '_groundtruth.npy', 'rb'))
    candidate_authors = [a[2:-3] for a in  open(groundtruth_path + '_candidate-labels.txt').read().split('\n')][:-1]
    query_authors = [a[2:-3] for a in  open(groundtruth_path + '_query-labels.txt').read().split('\n')][:-1]

    candidates_df['authorID'] = candidates_df.authorSetID.apply(lambda a: q_c_mapping(a))
    candidates_df['source'] = ['candidates'] * len(candidates_df)
    queries_df['source'] = ['queries'] * len(queries_df)
                                                     
    all_df = pd.concat([candidates_df, queries_df])[["authorID", "fullText", "documentID", "source"]]

    return all_df, queries_df, candidates_df