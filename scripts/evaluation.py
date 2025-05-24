import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.quantization.methods.quant_wrapper import quantize_model
from pm_kvq.evaluation.eval_wrapper import evaluate_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model")
parser.add_argument("--output_path", type=str, help="Path to the output .jsonl file")
parser.add_argument("--benchmark", type=str, help="Benchmark name", default="aime", choices=["aime", "cmimc", "livecodebench"])
parser.add_argument("--version", type=str, help="Benchmark version", default="2024")
parser.add_argument("--seed", type=int, help="Random seed for the first response", default=42)
parser.add_argument("--start", type=int, help="Start problem index", default=0)
parser.add_argument("--end", type=int, help="End problem index", default=30)
parser.add_argument("--n_responses", type=int, help="Number of responses per problem", default=16)
parser.add_argument("--method", type=str, help="Number of responses per problem", default="original", choices=["original", "pm-kvq", "rtn", "kivi", "rotatekv", "mikv"])
args, unknown = parser.parse_known_args()

if args.method == "original":
    pass

elif args.method == "pm-kvq":
    parser.add_argument("--backend", help="Backend to implement PM-KVQ", type=str, default="fake", choices=["fake", "real"])
    parser.add_argument("--rep_scales", help="Path to reparameterization scales", type=str, default=None)
    parser.add_argument("--kv_budgets", help="Path to KV Cache budgets", type=float)
    parser.add_argument("--n_sink_token", help="Number of sink tokens", type=int, default=1)
    parser.add_argument("--n_sink_token_bits", help="Bit-width of sink tokens", type=int, default=16)
    parser.add_argument("--n_window_token", help="Number of tokens in sliding window", type=int, default=128)
    parser.add_argument("--n_window_token_bits", help="Bit-width of tokens in sliding window", type=int, default=16)
    parser.add_argument("--n_init_kv_bits", help="Initial bit-width of KV Cache", type=int, default=16)

elif args.method == "rtn":
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

elif args.method == "kivi":
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

elif args.method == "rotatekv":
    parser.add_argument("--k_reorder_indices", help="Path to reorder indices", type=str, default=None)
    parser.add_argument("--k_had_dim", help="Dimension of head-wise hadamard, default to -1", type=int, default=-1)
    parser.add_argument("--n_pivot_token", help="Number of pivot tokens", type=int, default=20)
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

elif args.method == "mikv":
    parser.add_argument("--n_sink_tokens", help="Number of sink tokens", type=int, default=0)
    parser.add_argument("--k_bits", type=int, help="Bit-width of Key Cache", required=True)
    parser.add_argument("--v_bits", type=int, help="Bit-width of Value Cache", required=True)

else:
    raise NotImplementedError

args = parser.parse_args()
args_dict = vars(args)
method_kwargs = {key: args_dict[key] for key in args_dict if key not in ["model_path", "output_path", "seed", "benchmark", "version", "start", "end", "n_responses", "method"]}
evaluate_kwargs = {key: args_dict[key] for key in args_dict if key in ["output_path", "seed", "version", "start", "end", "n_responses"]}
generation_kwargs = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 32768, "do_sample": True}

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

quantize_model(model, args.method, method_kwargs)
evaluate_model(model, tokenizer, args.benchmark, evaluate_kwargs, generation_kwargs)
