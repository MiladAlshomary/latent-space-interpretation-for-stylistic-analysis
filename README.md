# Style Generation Pipeline

Example Usage:
```
python generate_styles.py --data-dir data/example_data.jsonl 
                          --generator-model llama3 
                          --shortener-model openai:gpt-3.5-turbo
                          --style-threshold 2
                          --max-new-tokens 512 
                          --device 2 
```

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
 - An example dataset is provided in `data/example_data.jsonl`. Please follow this format.
 - Setting `style-threshold` too low for small datasets may result in an empty filtered style description list.
 - The style generation and shortening steps use `llama3-8b` by default. To change the model, specify it with the `--generator-model` and `--shortener-model` arguments (see DOCSTRING in `generate_styles.py`).
 - Ensure your HuggingFace API key and OpenAI API key are included in `keys.json` and `backbones/openai_gpt/keys.json`.