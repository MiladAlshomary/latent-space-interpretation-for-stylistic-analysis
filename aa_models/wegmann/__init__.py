import numpy as np

from transformers import AutoModel, AutoTokenizer


class WEGMANN:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "AnnaWegmann/Style-Embedding", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            "AnnaWegmann/Style-Embedding", trust_remote_code=True
        )

    def encode(self, sentences, batch_size, convert_to_numpy=True, max_length=512):
        """
        Wrapper
        """
        batched_outputs = []

        for i in range(0, len(sentences), batch_size):
            texts_chunk = sentences[i : i + batch_size]

            # We have one document per author
            episode_length = 1
            tokenized_text = self.tokenizer(
                texts_chunk,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # inputs size: (batch_size, episode_length, max_token_length)
            tokenized_text["input_ids"] = tokenized_text["input_ids"].reshape(
                -1, episode_length, max_length
            )
            tokenized_text["attention_mask"] = tokenized_text["attention_mask"].reshape(
                -1, episode_length, max_length
            )

            out = self.model(**tokenized_text)
            batched_outputs.append(
                out.squeeze().detach().numpy()
                if convert_to_numpy
                else out.squeeze().detach()
            )

        return np.vstack(batched_outputs)
