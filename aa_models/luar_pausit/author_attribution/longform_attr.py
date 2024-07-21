from transformers import AutoTokenizer, TFLongformerModel
import sys
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from glob import glob
from typing import Tuple
import os
import numpy as np
from statistics import mode
import tensorflow as tf


def get_file_names(input_dir: str) -> Tuple[str, str]:
    queries_fname = glob(os.path.join(input_dir, "*queries*"))[0]
    candidates_fname = glob(os.path.join(input_dir, "*candidates*"))[0]
    return queries_fname, candidates_fname


def load_data(input_dir):
    queries_fname, candidates_fname = get_file_names(input_dir)

    prefix = queries_fname.split(os.sep)[-1]
    prefix = prefix[: prefix.find("_TA2")]

    candidates = pd.read_json(candidates_fname, lines=True)
    queries = pd.read_json(queries_fname, lines=True)

    q_identifier = "authorIDs"
    queries[q_identifier] = queries[q_identifier].apply(lambda x: x[0])
    c_identifier = "authorSetIDs"
    candidates[c_identifier] = candidates[c_identifier].apply(lambda x: x[0])

    X = queries["fullText"].tolist()
    Y = queries["authorIDs"].tolist()
    # X = candidates["fullText"].tolist()
    # Y = candidates["authorSetIDs"].tolist()
    X, Y = clean_data(X, Y)
    X_test = candidates["fullText"].tolist()
    Y_test = candidates["authorSetIDs"].tolist()
    # X_test = queries["fullText"].tolist()
    # Y_test = queries["authorIDs"].tolist()
    # Y_test = [mapping[m] for m in candidates["authorSetIDs"].tolist()]
    X_test, _ = clean_data(X_test, Y_test)
    le = preprocessing.LabelEncoder()
    Y = tf.keras.utils.to_categorical(le.fit_transform(Y))
    labels = le.classes_.tolist()
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    print(len(X), Y.shape)
    return X, Y, X_test, queries, candidates, prefix, labels


def clean_data(X, Y, max_len=1000000):

    if max_len is None:
        return X, Y

    X_clean = []
    Y_clean = []
    for i in range(len(X)):
        current = X[i]
        if any(char.isalnum() for char in current):
            if len(current) > max_len:
                current = current[:max_len]
            X_clean.append(current)
            Y_clean.append(Y[i])

    return X_clean, Y_clean


def build_model(max_len, num_labels, verbose=True):

    # tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    transform = TFLongformerModel.from_pretrained("allenai/longformer-base-4096")
    # X = tokenizer(
    #     text=X,
    #     add_special_tokens=True,
    #     max_length=max_len,
    #     truncation=True,
    #     padding='max_length',
    #     return_tensors='tf',
    #     return_token_type_ids = False,
    #     return_attention_mask = True,
    #     verbose = True)
    # X_val = tokenizer(
    #     text=X_val,
    #     add_special_tokens=True,
    #     max_length=max_len,
    #     truncation=True,
    #     padding='max_length',
    #     return_tensors='tf',
    #     return_token_type_ids = False,
    #     return_attention_mask = True,
    #     verbose = True)
    # X_test =tokenizer(
    #     text=X_test,
    #     add_special_tokens=True,
    #     max_length=max_len,
    #     truncation=True,
    #     padding='max_length',
    #     return_tensors='tf',
    #     return_token_type_ids = False,
    #     return_attention_mask = True,
    #     verbose = True)

    # if verbose:
    #     print(X['input_ids'].shape)
    #     print(X_val['input_ids'].shape)
    #     print(X_test['input_ids'].shape)

    input_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="input_ids"
    )
    input_mask = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="attention_mask"
    )
    embeddings = transform(input_ids, attention_mask=input_mask)[0]
    out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
    out = tf.keras.layers.Dense(256, activation="relu")(out)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.Dense(128, activation="relu")(out)
    y = tf.keras.layers.Dense(num_labels, activation="sigmoid")(out)
    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-05,  # this learning rate is for bert model , taken from huggingface website
        epsilon=1e-08,
        clipnorm=1.0,
    )
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metric = [tf.keras.metrics.CategoricalAccuracy("balanced_accuracy"), "accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    if verbose:
        print(model.summary())

    return model


def split_tokenize(X, Y, X_test, max_len):

    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

    X_train = tokenizer(
        text=X_train,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True,
    )
    X_val = tokenizer(
        text=X_val,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True,
    )
    X_test_cpy = tokenizer(
        text=X_test,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True,
    )
    return X_train, X_val, Y_train, Y_val, X_test_cpy


def apply_srs(input_dir, output_dir, run_id):

    X, Y, X_test, queries, candidates, prefix, labels = load_data(input_dir)

    # Check for GPU availability
    if False and tf.config.list_physical_devices("GPU"):
        # Configure TensorFlow to use GPU
        gpus = tf.config.experimental.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    with tf.device("/GPU:1"):
        num_labels = Y.shape[1]
        max_len = 1024
        # if len(sys.argv) > 1:
        #     max_len = int(sys.argv[1])
        model = build_model(num_labels=num_labels, max_len=max_len)

        log_callback = tf.keras.callbacks.CSVLogger("training.log")
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='longform_cp', save_weights_only=True, verbose=1)
        # es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=0.01, mode='max')

        for i in range(5):
            X_train, X_val, Y_train, Y_val, _ = split_tokenize(
                X=X, Y=Y, X_test=X_test, max_len=max_len
            )
            model.fit(
                x={
                    "input_ids": X_train["input_ids"],
                    "attention_mask": X_train["attention_mask"],
                },
                y=Y_train,
                validation_data=(
                    {
                        "input_ids": X_val["input_ids"],
                        "attention_mask": X_val["attention_mask"],
                    },
                    Y_val,
                ),
                callbacks=[log_callback],  # es_callback], #cp_callback
                epochs=5,
                batch_size=8,
            )
        X_test = _
        results = model.predict(
            x={
                "input_ids": X_test["input_ids"],
                "attention_mask": X_test["attention_mask"],
            },
            batch_size=16,
        )

        predictions = dict()
        count = dict()
        for i, row in candidates.iterrows():
            if row["authorSetIDs"] not in predictions:
                predictions[row["authorSetIDs"]] = np.zeros(len(labels))
                count[row["authorSetIDs"]] = 0
            predictions[row["authorSetIDs"]] = np.add(
                predictions[row["authorSetIDs"]], results[i]
            )
            count[row["authorSetIDs"]] += 1
        authorIDs = labels
        authorSetIDs = list(set(candidates["authorSetIDs"].tolist()))
        output_matrix = np.zeros((len(authorIDs), len(authorSetIDs)))
        for authorID, prediction in predictions.items():
            authorSetIDs_index = authorSetIDs.index(authorID)
            output_matrix[:, authorSetIDs_index] = prediction / count[authorID]

        # for i, row in queries.iterrows():
        #     if row["authorIDs"] not in predictions:
        #         predictions[row["authorIDs"]] = np.zeros(len(labels))
        #         count[row["authorIDs"]] = 0
        #     predictions[row["authorIDs"]] = np.add(predictions[row["authorIDs"]], results[i])
        #     count[row["authorIDs"]] += 1
        # authorIDs = list(set(queries["authorIDs"].tolist()))
        # authorSetIDs = labels
        # output_matrix = np.zeros((len(authorIDs), len(authorSetIDs)))
        # for authorID, prediction in predictions.items():
        #     authorIDs_index = authorIDs.index(authorID)
        #     output_matrix[authorIDs_index] = (prediction/count[authorID])

        save_output_files(
            prefix, output_dir, run_id, output_matrix, authorIDs, authorSetIDs
        )


def save_output_files(
    prefix, output_dir, run_id, output_matrix, authorIDs, authorSetIDs
):
    output_array_path = os.path.join(
        output_dir, f"{prefix}_TA2_query_candidate_attribution_scores_{run_id}.npy"
    )
    output_query_labels_path = os.path.join(
        output_dir,
        f"{prefix}_TA2_query_candidate_attribution_query_labels_{run_id}.txt",
    )
    output_candidate_labels_path = os.path.join(
        output_dir,
        f"{prefix}_TA2_query_candidate_attribution_candidate_labels_{run_id}.txt",
    )
    with open(output_array_path, "wb") as f:
        np.save(f, output_matrix)

    with open(output_query_labels_path, "w") as f:
        for line in authorIDs:
            f.write(f"('{line}',)\n")

    with open(output_candidate_labels_path, "w") as f:
        for line in authorSetIDs:
            f.write(f"('{line}',)\n")
