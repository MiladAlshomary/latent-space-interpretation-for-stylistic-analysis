## Generating explanations for phase_1

export TRANSFORMERS_CACHE="/mnt/swordfish-pool2/milad/hf-cache-new"
export HF_DATASETS_CACHE="/mnt/swordfish-pool2/milad/hf-cache-new"
export HF_HOME="/mnt/swordfish-pool2/milad/hf-cache-new"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

### Generating styels:

- To generate for the sample of data

python generate_styles.py --data-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_1/raw_data_sample.jsonl           --generator-model llama3           --shortener-model openai:gpt-3.5-turbo           --style-threshold 2           --max-new-tokens 512           --device 0 --datadreamer_path /mnt/swordfish-pool2/milad/hiatus-data/phase_1/datadreamer

- To generate for all training and dev authors in the data
python generate_styles.py --data-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explainability/training_candidates_and_queries.jsonl           --generator-model llama3           --shortener-model openai:gpt-3.5-turbo           --style-threshold 2           --max-new-tokens 512           --device 0 1 5 --datadreamer_path /mnt/swordfish-pool2/milad/hiatus-data/phase_1/datadreamer

- Afterwards, I moved the output of the command to `explainability` folder

### Clustering documents:

- For the data sample
yes | python cluster_documents.py --train-dir "/mnt/swordfish-pool2/milad/hiatus-data/phase_1/training_authors_sample.json" --test-dir "/mnt/swordfish-pool2/milad/hiatus-data/phase_1/valid_authors_sample.json" --save-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explanation/ --model aa_model-luar  --style-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explanation/filtered/refined_and_aggregated_features_final.csv --top_k_feats 10 --eps 0.16

- For all authors in training and dev
yes | python cluster_documents.py --train-dir "/mnt/swordfish-pool2/milad/hiatus-data/phase_1/training_authors.json" --test-dir "/mnt/swordfish-pool2/milad/hiatus-data/phase_1/valid_authors.json" --save-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explainability/ --model aa_model-luar  --style-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explainability/refined_and_aggregated_features_final.csv --top_k_feats 10 --eps 0.16

### Generate explanations

yes | python generate_explanations.py --inter-path /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explainability/interpretable_space.pkl --input-path /mnt/swordfish-pool2/milad/hiatus-data/phase_1/valid_authors.json --output-path /mnt/swordfish-pool2/milad/hiatus-data/phase_1/explainability/test_authors_explained.json --top-k 5 --top-c 3 --model aa_model-luar

## Generating styles for phase_2

### Generate the styles:

export TRANSFORMERS_CACHE="/mnt/swordfish-pool2/milad/hf-cache-new"
export HF_DATASETS_CACHE="/mnt/swordfish-pool2/milad/hf-cache-new"
export HF_HOME="/mnt/swordfish-pool2/milad/hf-cache-new"
export CUDA_VISIBLE_DEVICES="5,6"

python generate_styles.py --data-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/all_documents_in_cross_genre.jsonl \
                          --generator-model llama3 \
                          --shortener-model openai:gpt-3.5-turbo \
                          --style-threshold 2 \
                          --max-new-tokens 512 \
                          --device 0 1 \


### Perform clustering according using the AA model

python cluster_documents.py --train-dir "/mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/train_authors.json" --test-dir "/mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/test_authors.json" --save-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/ --model aa_model-luar --eps-threshold 0.02 --style-dir /mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/filtered/refined_and_aggregated_features_final.csv 


### Generate explatanions

python generate_explanations.py --inter-space /mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/interpretable_space.pkl --input-path /mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/test_authors.json --output-path /mnt/swordfish-pool2/milad/hiatus-data/phase_2/explainability/test_authors.json