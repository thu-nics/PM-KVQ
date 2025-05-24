import os
import json
from tqdm import tqdm

import torch

from pm_kvq.utils.chatbot import chat
from pm_kvq.evaluation.eval_livecodebench.livecodebench import load_code_generation_dataset

DEFAULT_CODE_GENERATION_DATASET_PATH = "livecodebench/code_generation_lite"


def eval_livecodebench_code_generation(model, tokenizer, dataset_path=DEFAULT_CODE_GENERATION_DATASET_PATH, version="v6", n_responses=1, record=True, output_path=None, start=None, end=None, seed=42, **kwargs):
    json_data = {}
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = load_code_generation_dataset(dataset_path, release_version=version)["test"]
    if start is not None and end is not None:
        dataset = dataset.select(range(start, end))

    for problem_id, sample in enumerate(tqdm(dataset)):
        for i in range(n_responses):
            torch.manual_seed(seed + i)
            prompt = f"### Question:\n{sample['question_content']}\n\n{sample['format_prompt']}" + "### Answer: (use the provided format with backticks)\n\n"
            response, length = chat(model, tokenizer, text=prompt, print_response=False, return_len=True, **kwargs)
            if record:
                json_data[f"{sample['question_id']}.{i}"] = {
                    "seed": seed + i,
                    "response": response,
                    "input_len": length[0],
                    "output_len": length[1],
                }
                with open(output_path, "w") as f:
                    json.dump(json_data, f, indent=4)
