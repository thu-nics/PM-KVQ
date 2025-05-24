import re
import os
import json
from tqdm import tqdm
import pandas as pd

import torch
from datasets import Dataset

from pm_kvq.utils.chatbot import chat

DEFAULT_DATASET_PATH = "datasets/aime/"


def eval_aime(model, tokenizer, dataset_path=DEFAULT_DATASET_PATH, version=2024, n_responses=1, record=True, output_path=None, start=None, end=None, seed=42, **kwargs):
    json_data = {}
    problems, answers, ids = [], [], []
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for subset in [f"aime_{version}_I", f"aime_{version}_II"]:
        # load question and answer
        for i in range(1, 16):
            problem_file = os.path.join(dataset_path, subset, "problems", f"{i}.tex")
            with open(problem_file, "r") as f:
                problems.append(f.read())
                ids.append(f"{subset}.{i}")

        answer_file = os.path.join(dataset_path, subset, "answers.csv")
        subset_answers = pd.read_csv(answer_file)["answer"].tolist()
        answers += [str(answer) for answer in subset_answers]
    dataset = Dataset.from_dict({"problem": problems, "answer": answers, "id": ids})
    if start is not None and end is not None:
        dataset = dataset.select(range(start, end))

    for problem_id, sample in enumerate(tqdm(dataset)):
        problem = sample["problem"]
        prompt = f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        for i in range(n_responses):
            torch.manual_seed(seed + i)
            response, length = chat(model, tokenizer, text=prompt, print_response=False, return_len=True, **kwargs)
            response_answer = math_postprocess(response)
            judgement = judge(response_answer, sample["answer"])
            if record:
                json_data[f"{sample['id']}.{i}"] = {
                    "seed": seed + i,
                    "response": response,
                    "response_answer": response_answer,
                    "gold": sample["answer"],
                    "judgement": judgement,
                    "input_len": length[0],
                    "output_len": length[1],
                }
                with open(output_path, "w") as f:
                    json.dump(json_data, f, indent=4)
    acc = calculate_acc(output_path)
    return acc


def judge(str1, str2):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    strip_string_func = _strip_string

    try:
        ss1 = strip_string_func(str1)
        ss2 = strip_string_func(str2)
        if ss1 == ss2:
            return True
        ss1 = normalize_final_answer(ss1)
        ss2 = normalize_final_answer(ss2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    try:
        ss1 = normalize_final_answer(str1)
        ss2 = normalize_final_answer(str2)
        if ss1 == ss2:
            return True
    except Exception:
        pass

    return str1 == str2


def calculate_acc(output_path):
    with open(output_path, "r") as f:
        json_data = json.load(f)
    judgements = [json_data[key]["judgement"] for key in json_data]
    return sum(judgements) / len(judgements)


def math_postprocess(text: str) -> str:

    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    for maybe_ans in text.split("."):
        # if 'final answer' in maybe_ans.lower():
        if re.search("final answer|answer is", maybe_ans.lower()):
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split(".")[0])


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match("^\{(.*)\}$", answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # final_answer = final_answer.split('=')[-1]
    SUBSTITUTIONS = [
        ("an ", ""),
        ("a ", ""),
        (".$", "$"),
        ("\\$", ""),
        (r"\ ", ""),
        (" ", ""),
        ("mbox", "text"),
        (",\\text{and}", ","),
        ("\\text{and}", ","),
        ("\\text{m}", "\\text{}"),
        ("\\le", "<"),
    ]
    REMOVED_EXPRESSIONS = [
        "square",
        "ways",
        "integers",
        "dollars",
        "mph",
        "inches",
        "ft",
        "hours",
        "km",
        "units",
        "\\ldots",
        "sue",
        "points",
        "feet",
        "minutes",
        "digits",
        "cents",
        "degrees",
        "cm",
        "gm",
        "pounds",
        "meters",
        "meals",
        "edges",
        "students",
        "childrentickets",
        "multiples",
        "\\text{s}",
        "\\text{.}",
        "\\text{\ns}",
        "\\text{}^2",
        "\\text{}^3",
        "\\text{\n}",
        "\\text{}",
        r"\mathrm{th}",
        r"^\circ",
        r"^{\circ}",
        r"\;",
        r",\!",
        "{,}",
        '"',
        "\\dots",
        "\n",
        "\r",
        "\f",
    ]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(\\text\{)\((.*?)\)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    assert "\n" not in final_answer
    assert "\r" not in final_answer
    assert "\f" not in final_answer
    if len(re.findall(r"finalansweris(.*)", final_answer)) > 0:
        final_answer = re.findall(r"finalansweris(.*)", final_answer)[-1]

    if len(re.findall(r"answer?is:?(.*)", final_answer)) > 0:
        final_answer = re.findall(r"answer?is:?(.*)", final_answer)[-1]

    if len(re.findall(r"oxed\{(.*?)\}", final_answer)) > 0:
        final_answer = re.findall(r"oxed\{(.*?)\}", final_answer)[-1]

    if len(re.findall(r"\$(.*?)\$", final_answer)) > 0:
        final_answer = re.findall(r"\$(.*?)\$", final_answer)[-1]
    final_answer = final_answer.strip()
    if "rac" in final_answer and "\\frac" not in final_answer:
        final_answer = final_answer.replace("rac", "\\frac")

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def _strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
    # add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace('"', "")

    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple
    # cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string
