import pandas as pd


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
    data_df = pd.read_json(data_path, lines=True)
    data_df["authorID"] = data_df["authorIDs"].str[0]
    if group_by_author:
        data_df = (
            data_df.groupby("authorID")
            .agg({"fullText": list, "documentID": list})
            .reset_index()
        )

    return data_df
