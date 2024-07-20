import os
import json
import random
import pickle as pkl
import multiprocessing

import openai
from munch import Munch
from tqdm import tqdm
from openai import OpenAI

from utils import partition


# Create client upon import
client = None


def get_prompt_response(instruction, instance, model, temperature):
    chat_models = ["gpt-3.5-turbo", "gpt-4"]
    chat_model = False
    for chat_m in chat_models:
        if chat_m in model:
            chat_model = True

    if chat_model:
        response = (
            client.chat.completions.create(
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": instance},
                ],
                model=model,
                temperature=temperature,
            )
            .choices[0]
            .message.content
        )
    else:
        response = (
            client.completions.create(
                model=model,
                prompt=instance,
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            .choices[0]
            .text
        )

    return response


def query_worker(model, inputs, process_id, save_dir, lock, instruction, temperature):
    """ """

    def query_loop(model, instruction, instance, temperature):
        """ """
        # Edge case: If an empty string is passed, return an empty string
        if instance == "":
            return ""

        server_error = False

        response = None
        while response == None:
            try:
                response = get_prompt_response(
                    instruction, instance, model, temperature
                )

            except openai.InternalServerError:
                # If first time encountering error, change model to long-context version
                if server_error == False:
                    server_error = True
                    model += "-16k"
                # Otherwise, adding prompt did not help, so exit
                else:
                    print(
                        "[InternalServerError]: Likely generated invalid Unicode output."
                    )
                    print(instance)
                    exit()
            except openai.BadRequestError:
                print(instance)
                print(
                    "[BadAPIRequest] Likely input too long or invalid settings (e.g. temperature > 2)."
                )
                exit()
            except openai.Timeout:
                continue

        return response

    with lock:
        bar = tqdm(
            desc=f"Process {process_id+1}",
            total=len(inputs),
            position=process_id + 1,
            leave=False,
        )

    # If partially populated results file exists, load and continue
    responses = pkl.load(open(save_dir, "rb")) if os.path.exists(save_dir) else list()
    start_index = len(responses)

    for instance in inputs[start_index:]:
        with lock:
            bar.update(1)

        response = query_loop(model, instruction, instance, temperature)
        responses.append(response)

        pkl.dump(responses, open(save_dir, "wb"))

    with lock:
        bar.close()

    return responses


class OpenAIModel:
    def __init__(self, model_name):
        """ """
        global client
        client = OpenAI(
            api_key=random.choice(
                json.load(
                    open(os.path.join(os.path.dirname(__file__), "keys.json"), "r")
                )
            )
        )

        args = Munch.fromYAML(
            open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r")
        )

        self.model = model_name
        self.num_processes = args.num_processes

    def infer_batch(self, inputs, save_dir, instruction, temperature=0):
        """ """
        # Partition instances into
        paritioned_inputs = partition(inputs, self.num_processes)

        # Start multiprocess instances
        lock = multiprocessing.Manager().Lock()
        pool = multiprocessing.Pool(processes=self.num_processes)

        worker_results = []
        for process_id in range(len(paritioned_inputs)):
            async_args = (
                self.model,
                paritioned_inputs[process_id],
                process_id,
                save_dir.replace(
                    f".{save_dir.split('.')[-1]}",
                    f"-process={process_id}.{save_dir.split('.')[-1]}",
                ),
                lock,
                instruction,
                temperature,
            )

            # Run each worker
            worker_results.append(pool.apply_async(query_worker, args=async_args))

        pool.close()
        pool.join()

        responses = []
        for worker_result in worker_results:
            responses += worker_result.get()

        return responses
