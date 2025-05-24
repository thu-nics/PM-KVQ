import os
import json

DEFAULT_CODE_GENERATION_DATASET_PATH = "livecodebench/code_generation_lite"


def judge_aime(responses_dir, version):
    test_files = [os.path.join(responses_dir, path) for path in os.listdir(responses_dir)]
    test_data = {}
    for test_file in test_files:
        with open(test_file, "r") as f:
            data = json.load(f)
            for k in data.keys():
                data[k].pop("response")
        test_data.update(data)

    if len(test_data) == 16 * 30:
        print("complete test data")
    else:
        print(f"incomplete test data:{len(test_data)}/480")

    ids = [f"aime_{version}_I.{i}" for i in range(1, 16)] + [f"aime_{version}_II.{i}" for i in range(1, 16)]
    detailed_pass_at_1 = [0] * 30
    detailed_voting_judgements = [None] * 30
    avg_length = [0] * 30

    for i, idx in enumerate(ids):
        test_keys = [key for key in test_data.keys() if key.startswith(f"{idx}.")]
        if len(test_keys) == 0:
            continue
        gold = test_data[test_keys[0]]["gold"]
        judgement = [test_data[test_key]["judgement"] for test_key in test_keys]
        detailed_pass_at_1[i] = sum(judgement) / len(judgement) * 100
        length = [test_data[test_key]["input_len"] + test_data[test_key]["output_len"] for test_key in test_keys]
        avg_length[i] = sum(length) / len(length)

        response_answer = [test_data[test_key]["response_answer"] for test_key in test_keys]
        vote_answer = max(response_answer, key=response_answer.count)
        detailed_voting_judgements[i] = vote_answer == gold

    overall_judgements = [v["judgement"] for k, v in test_data.items()]
    overall_pass_at_1 = sum(overall_judgements) / len(overall_judgements) * 100
    overall_voting_acc = [x for x in detailed_voting_judgements if x is not None]
    overall_voting_acc = sum(overall_voting_acc) / len(overall_voting_acc) * 100

    return (
        overall_pass_at_1,
        overall_voting_acc,
        detailed_pass_at_1,
        detailed_voting_judgements,
        avg_length,
    )


def judge_cmimc(responses_dir, version):
    test_files = [os.path.join(responses_dir, path) for path in os.listdir(responses_dir)]
    test_data = {}
    for test_file in test_files:
        with open(test_file, "r") as f:
            data = json.load(f)
            for k in data.keys():
                data[k].pop("response")
        test_data.update(data)

    if len(test_data) == 16 * 30:
        print("complete test data")
    else:
        print(f"incomplete test data:{len(test_data)}/480")

    ids = [f"cmimc_{version}.{i}" for i in range(1, 31)]
    detailed_pass_at_1 = [0] * 30
    detailed_voting_judgements = [None] * 30
    avg_length = [0] * 30

    for i, idx in enumerate(ids):
        test_keys = [key for key in test_data.keys() if key.startswith(f"{idx}.")]
        if len(test_keys) == 0:
            continue
        gold = test_data[test_keys[0]]["gold"]
        judgement = [test_data[test_key]["judgement"] for test_key in test_keys]
        detailed_pass_at_1[i] = sum(judgement) / len(judgement) * 100
        length = [test_data[test_key]["input_len"] + test_data[test_key]["output_len"] for test_key in test_keys]
        avg_length[i] = sum(length) / len(length)

        response_answer = [test_data[test_key]["response_answer"] for test_key in test_keys]
        vote_answer = max(response_answer, key=response_answer.count)
        detailed_voting_judgements[i] = vote_answer == gold

    overall_judgements = [v["judgement"] for k, v in test_data.items()]
    overall_pass_at_1 = sum(overall_judgements) / len(overall_judgements) * 100
    overall_voting_acc = [x for x in detailed_voting_judgements if x is not None]
    overall_voting_acc = sum(overall_voting_acc) / len(overall_voting_acc) * 100

    return (
        overall_pass_at_1,
        overall_voting_acc,
        detailed_pass_at_1,
        detailed_voting_judgements,
        avg_length,
    )


def judge_livecodebench(responses_dir, version):
    from pm_kvq.evaluation.eval_livecodebench.livecodebench import load_code_generation_dataset
    from pm_kvq.evaluation.eval_livecodebench.evaluator import score_code_generation

    test_files = [os.path.join(responses_dir, path) for path in os.listdir(responses_dir)]
    test_data = {}
    for test_file in test_files:
        with open(test_file, "r") as f:
            data = json.load(f)
        test_data.update(data)
    if len(test_data) == 175 * 4:
        print("complete test data")
    else:
        print(f"incomplete test data:{len(test_data)}/700")
    references = [k.split(".")[0] for k in test_data.keys()]
    predictions = [v["response"] for v in test_data.values()]

    dataset = load_code_generation_dataset(DEFAULT_CODE_GENERATION_DATASET_PATH, version)["test"]
    results = score_code_generation(
        dataset,
        predictions,
        references,
    )
    return results
