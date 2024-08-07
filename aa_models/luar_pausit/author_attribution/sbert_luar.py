from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
from typing import List, Dict, Optional, Union, Tuple
from collections import OrderedDict
import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models, evaluation
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import(
    import_from_string,
    batch_to_device,
    fullname,
    is_sentence_transformer_model,
    load_dir_path,
    load_file_path,
    save_to_hub_args_decorator,
    get_device_name,
)

import torch
import transformers

# sentence transformer version
__version__ = "2.5.0"

class LuarTransformer(models.Transformer):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Dict = {},
        cache_dir: Optional[str] = None,
        tokenizer_args: Dict = {},
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
    ):
        super().__init__(model_name_or_path, 512, model_args, cache_dir, tokenizer_args, do_lower_case, tokenizer_name_or_path)

        self.batch_size = 16
        self.author_level = True
        self.text_key = "fullText"
        self.token_max_length = 512
        self.document_batch_size = 32

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(
            self.get_config_dict(), self.auto_model.__class__.__name__
        )

    
    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        self.auto_model = AutoModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )

    def forward(self, features):
        # """Returns token_embeddings, cls_token"""
        # trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}
        # if "token_type_ids" in features:
        #     trans_features["token_type_ids"] = features["token_type_ids"]

        batch_size = self.batch_size
        identifier = "documentID"

        input_ids = features['input_ids']
        attention_mask = features['attention_mask']

        input_ids = input_ids.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        input_ids = input_ids.reshape((-1, 1, self.token_max_length))
        attention_mask = attention_mask.reshape((-1, 1, self.token_max_length))
        output_states = self.auto_model(input_ids, attention_mask, document_batch_size=self.document_batch_size)

        
        output_tokens = output_states


        features.update({"sentence_embedding": output_tokens, "attention_mask": features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                truncation=True,
                padding="max_length",
                max_length=self.token_max_length,
                return_tensors="pt"
            )
        )
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        #self.tokenizer.save_pretrained(output_path) # we don't save the tokenizer because its throwing an error. We load it from the original path 

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        # Don't allow configs to set trust_remote_code
        if "model_args" in config:
            config["model_args"].pop("trust_remote_code")

        config["model_args"]={'trust_remote_code':True}
        return LuarTransformer(model_name_or_path=input_path, tokenizer_name_or_path="rrivera1849/LUAR-MUD", **config)


class LuarSentenceTransformer(SentenceTransformer):

    def save(
        self,
        path: str,
        model_name: Optional[str] = None,
        create_model_card: bool = True,
        train_datasets: Optional[List[str]] = None,
    ):
        """
        Saves all elements for this seq. sentence embedder into different sub-folders

        :param path: Path on disc
        :param model_name: Optional model name
        :param create_model_card: If True, create a README.md with basic information about this model
        :param train_datasets: Optional list with the names of the datasets used to to train the model
        """
        if path is None:
            return

        os.makedirs(path, exist_ok=True)

        modules_config = []

        # Save some model info
        if "__version__" not in self._model_config:
            self._model_config["__version__"] = {
                "sentence_transformers": __version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            }

        with open(os.path.join(path, "config_sentence_transformers.json"), "w") as fOut:
            config = self._model_config.copy()
            config["prompts"] = self.prompts
            config["default_prompt_name"] = self.default_prompt_name
            json.dump(config, fOut, indent=2)

        # Save modules
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            if idx == 0 and isinstance(module, LuarTransformer):  # Save transformer model in the main folder
                model_path = path + "/"
            else:
                model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            modules_config.append(
                {"idx": idx, "name": name, "path": os.path.basename(model_path), "type": "author_attribution.sbert_luar.LuarTransformer" if type(module).__module__=="author_attribution.sbert_luar" else type(module).__module__}
            )

        with open(os.path.join(path, "modules.json"), "w") as fOut:
            json.dump(modules_config, fOut, indent=2)

        # Create model card
        if create_model_card:
            self._create_model_card(path, model_name, train_datasets)