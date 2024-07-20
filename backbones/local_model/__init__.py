import os
import pickle as pkl

import ollama
from tqdm import tqdm


class LocalModel:
    def __init__(self, model_name):
        """ """
        self.model = model_name

    def infer_batch(self, inputs, save_dir):
        """ """
        responses = (
            pkl.load(open(save_dir, "rb")) if os.path.exists(save_dir) else list()
        )
        start_index = len(responses)

        for instance in tqdm(
            inputs[start_index:], desc=self.model, ascii=True, leave=False
        ):
            if type(instance) == list:
                response = [
                    (
                        ollama.generate(model=self.model, prompt=instance_item)[
                            "response"
                        ]
                        if instance_item != ""
                        else ""
                    )
                    for instance_item in instance
                ]
            elif type(instance) == str:
                response = ollama.generate(model=self.model, prompt=instance)[
                    "response"
                ]

            responses.append(response)

            pkl.dump(responses, open(save_dir, "wb"))

        return responses
