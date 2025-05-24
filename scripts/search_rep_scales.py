import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.datasets.calib_dataset import get_calib_redpajama
from pm_kvq.quantization.methods.pm_kvq.smoothattention.searching_scales import search_rep_scales

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--effective_len", type=int, default=None)
parser.add_argument("--max_keys_path", type=str, default=None)
parser.add_argument("--grid", type=int, default=20)
parser.add_argument("--k_bits", type=int, default=-1)
parser.add_argument("--v_bits", type=int, default=-1)
parser.add_argument("--save_path", type=str, default=None)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

dataset = get_calib_redpajama(args.dataset_path, args.n_samples, args.seq_len, tokenizer)
max_keys = torch.load(args.max_keys_path)

rep_scales = search_rep_scales(
    model,
    k_config={"n_bits": args.k_bits, "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False},
    v_config={"n_bits": args.v_bits, "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False},
    dataset=dataset,
    max_keys=max_keys,
    effective_len=args.effective_len,
    save_path=args.save_path,
)
