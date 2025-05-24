import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.datasets.calib_dataset import get_calib_redpajama
from pm_kvq.quantization.methods.rotatekv.apply_rotatekv import get_k_reorder_indices

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--save_path", type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

calib_dataset = get_calib_redpajama(args.dataset_path, args.n_samples, args.seq_len, tokenizer)
max_keys = get_k_reorder_indices(model, calib_dataset, args.save_path)
