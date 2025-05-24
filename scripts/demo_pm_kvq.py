import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.quantization.methods.pm_kvq.apply_pmkvq import apply_fake_pmkvq, apply_real_pmkvq
from pm_kvq.utils.chatbot import chat

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model")

parser.add_argument("--backend", help="Backend to implement PM-KVQ", type=str, default="fake", choices=["fake", "real"])
parser.add_argument("--rep_scales", help="Path to reparameterization scales", type=str, default=None)
parser.add_argument("--kv_budgets", help="Path to KV Cache budgets", type=str)
parser.add_argument("--n_sink_token", help="Number of sink tokens", type=int, default=1)
parser.add_argument("--n_sink_token_bits", help="Bit-width of sink tokens", type=int, default=16)
parser.add_argument("--n_window_token", help="Number of tokens in sliding window", type=int, default=128)
parser.add_argument("--n_window_token_bits", help="Bit-width of tokens in sliding window", type=int, default=16)
parser.add_argument("--n_init_kv_bits", help="Initial bit-width of KV Cache", type=int, default=16)

args = parser.parse_args()
args_dict = vars(args)
method_kwargs = {key: args_dict[key] for key in args_dict if key not in ["model_path"]}
generation_kwargs = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768, "do_sample": True}

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

with open("datasets/aime/aime_2024_I/problems/8.tex") as f:
    text = f.read()
text = f"{text}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

backend = method_kwargs.pop("backend")
if backend == "fake":
    apply_fake_pmkvq(model, **method_kwargs)
elif backend == "real":
    apply_real_pmkvq(model, **method_kwargs)

torch.manual_seed(42)
chat(model, tokenizer, text, print_response=True, **generation_kwargs)
