import os
import sys
import json

import torch
from transformers import QuantoConfig
from transformers import BitsAndBytesConfig
from datadreamer.llms import HFTransformers, ParallelLLM, OpenAI

from utils import check_model_name_format


def get_datadreamer_backbone(model_name, device):
    if model_name == "llama3":
        if len(device) == 1:
            device = device[0]
            return HFTransformers(
                "meta-llama/Meta-Llama-3-8B-Instruct",
                quantization_config=QuantoConfig(weights="int8"),
                device=device,
                device_map="cuda",
            )

        else:
            return ParallelLLM(
                *[
                    HFTransformers(
                        "meta-llama/Meta-Llama-3-8B-Instruct",
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                        device=d,
                        dtype=torch.bfloat16,
                    )
                    for d in device
                ]
            )

    elif model_name == "mistral":
        if len(device) == 1:
            device = device[0]
            return HFTransformers(
                "mistralai/Mistral-7B-Instruct-v0.2",
                quantization_config=QuantoConfig(weights="int8"),
                device_map="cuda",
                device=device,
            )

        else:
            return ParallelLLM(
                *[
                    HFTransformers(
                        "mistralai/Mistral-7B-Instruct-v0.2",
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                        device=d,
                        dtype=torch.bfloat16,
                    )
                    for d in device
                ]
            )

    elif "openai" in model_name:
        api_key = json.load(
            open(os.path.join(os.path.dirname(__file__), "../keys.json"), "r")
        )["openai"]

        # Check formats and OpenAI API key is properly initialized
        check_model_name_format(model_name)
        assert api_key != ""

        return OpenAI(model_name=model_name.split(":")[-1], api_key=api_key)


def get_model(model_name):
    """
    Load and return the appropriate model object based on the given model name.

    This function dynamically adds the current directory to the system path,
    checks the provided model name, and imports and initializes the corresponding
    model class.

    Parameters:
    model (str): The name of the model to load. Expected values are substrings
                 that indicate the type of model, such as "openai" for OpenAI models
                 or other strings for local Llama2 models.

    Returns:
    object: An instance of the specified model class, either OpenAIModel or Llama2Model.
    """
    if os.path.dirname(__file__) not in sys.path:
        sys.path.append(os.path.dirname(__file__))

    if "openai" in model_name:
        from openai_gpt import OpenAIModel

        model = OpenAIModel(model_name.split(":")[-1])

    else:
        from local_model import LocalModel

        model = LocalModel(model_name)

    return model
