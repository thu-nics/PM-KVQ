# Copyright (c) 2024, LiveCodeBench and its contributors.
# Copyright (c) 2023, OpenCompass and its contributors.

import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from datasets import DatasetDict, load_dataset, load_from_disk

from pm_kvq.evaluation.eval_livecodebench.prompts import SelfRepairPromptConstants  # noqa: F401, F403
from pm_kvq.evaluation.eval_livecodebench.prompts import TestOutputPromptConstants  # noqa: F401, F403
from pm_kvq.evaluation.eval_livecodebench.prompts import CodeGenerationPromptConstants, get_generic_question_template_answer_self_repair, get_generic_question_template_test_completion, make_code_execution_prompt


def load_code_generation_dataset(
    path,
    release_version: str = "release_v6",
    start_date: str = None,
    end_date: str = None,
):

    def transform(item):
        # Define the dataitem mapping logic

        # starter_code
        if item["starter_code"]:
            format_prompt = f"### Format: {CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"  # noqa: E501
            format_prompt += f"```python\n{item['starter_code']}\n```\n\n"  # noqa: Q000, E501
        else:
            format_prompt = f"### Format: {CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"  # noqa: E501
            format_prompt += "```python\n# YOUR CODE HERE\n```\n\n"

        item["format_prompt"] = format_prompt

        # load test cases
        public_test_cases = item["public_test_cases"]
        public_test_cases = json.loads(item["public_test_cases"])

        private_test_cases = item["private_test_cases"]
        try:
            private_test_cases = json.loads(item["private_test_cases"])
        except Exception as e:  # noqa: F841
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode("utf-8"))))  # type: ignore
            )  # type: ignore

        # load metadata
        metadata = json.loads(item["metadata"])
        evaluation_sample = json.dumps(
            {
                "inputs": [t["input"] for t in public_test_cases + private_test_cases],
                "outputs": [t["output"] for t in public_test_cases + private_test_cases],
                "fn_name": metadata.get("func_name", None),
            }
        )
        item["evaluation_sample"] = evaluation_sample

        return item

    dataset = load_dataset(path, split="test", version_tag=release_version, trust_remote_code=True)  # 'livecodebench/code_generation_lite'

    dataset = dataset.map(transform)

    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = dataset.filter(lambda e: p_start_date <= datetime.fromisoformat(e["contest_date"]))  # noqa: E501
    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = dataset.filter(lambda e: datetime.fromisoformat(e["contest_date"]) <= p_end_date)  # noqa: E501

    return DatasetDict({"test": dataset, "train": dataset})


def load_code_execution_dataset(
    path,
    cot: bool = False,
    # release_version: str = "release_v1"
):
    # path = get_data_path(path, local_mode=local_mode)

    def transform(item):
        code, input = item["code"], item["input"]
        prompt = make_code_execution_prompt(code, input, cot=cot)

        item["prompt"] = prompt

        evaluation_sample = json.dumps({"code": item["code"], "input": item["input"], "output": item["output"]})
        item["evaluation_sample"] = evaluation_sample

        return item

    dataset = load_dataset(path, split="test")  # 'livecodebench/execution-v2'
    dataset = dataset.map(transform)

    return DatasetDict({"test": dataset, "train": dataset})


def load_output_prediction_dataset(
    path,
    # release_version: str = "release_v1"
):

    def transform(item):
        question_content = item["question_content"]
        starter_code = item["starter_code"]
        test = json.loads(item["test"])

        testcase_input = test[0]["input"]
        testcase_output = test[0]["output"]

        item["testcase_input"] = testcase_input
        item["testcase_output"] = testcase_output

        item["prompt"] = get_generic_question_template_test_completion(question_content=question_content, starter_code=starter_code, testcase_input=testcase_input)

        evaluation_sample = json.dumps({"input": item["question_content"], "output": json.loads(item["test"])[0]["output"]})
        item["evaluation_sample"] = evaluation_sample

        return item

    dataset = load_dataset(path, split="test", trust_remote_code=True)
    dataset = dataset.map(transform)

    return DatasetDict({"test": dataset, "train": dataset})
