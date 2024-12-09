# Introduction

This repository contain the code for our paper Latent Space Interpretation for Stylistic Analysis and Explainable Authorship Attribution (https://arxiv.org/abs/2409.07072), accepted at COLIING-25.

# Style Generation Pipeline

Example usage for generating styles:
```
python generate_styles.py --data-dir datasets/example_data.jsonl 
                          --generator-model llama3 
                          --shortener-model openai:gpt-3.5-turbo
                          --style-threshold 2
                          --max-new-tokens 512 
                          --device 2 
```
Example usage for clustering:
```
python cluster_documents.py --train-dir datasets/example_data.jsonl 
                            --test-dir datasets/example_data.jsonl 
                            --save-dir results/
```
`cluster_documents.py` must be run after `generate_styles.py` to construct the interpretable space, where each basis is mapped to a style distribution.

Example usage for generating style explanations:
```
python generate_explanations.py --inter-space <path to the interpretable space generated from cluster_documents.py code> 
                                --input-path <path to the dataframe containing the documents to be explained> 
                                --output-path <path to where to save the output>

```
`cluster_documents.py` must be run after `generate_styles.py` to construct the interpretable space, where each basis is mapped to a style distribution.

## Requirements
Install the necessary libraries using the provided `requirements.txt`:
```
pip install -r requirements.txt
```
Alternatively, you can manually install them:
```
pandas
mutual-implication-score
spacy
python-levenshtein
numpy
plotly
tqdm
nltk
scikit-learn
datasets
huggingface-hub
datadreamer
torch
transformers
ollama
openai
munch
```

## Notes
 - An example dataset is provided in `datasets/example_data.jsonl`. Please follow this format.

### Style Generator
 - Setting `style-threshold` too low for small datasets may result in an empty filtered style description list.
 - The style generation and shortening steps use `llama3-8b` by default. To change the model, specify it with the `--generator-model` and `--shortener-model` arguments (see DOCSTRING in `generate_styles.py`).
 - Ensure your HuggingFace API key and OpenAI API key are included in `keys.json` and `backbones/openai_gpt/keys.json`.

### Clustering (POI Identification)
- The default clustering method used is DBSCAN with cosine dissimilarity as the metric.
- Points of Interest (POIs) are identified by iterating through a range of $\epsilon$ values for DBSCAN and selecting the value where the performance gain is minimal across all metrics (EER, AP, NDCG). The default range is 0.01 to 2 with a step size of 0.01.
- Using a train and test dataset that is too small may result in degenerate clustering outcomes due to the nature of DBSCAN.

### Generating style explanations for documents
- Generate explanations for the style of documents