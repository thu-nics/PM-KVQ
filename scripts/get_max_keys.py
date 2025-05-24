import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.datasets.calib_dataset import get_calib_redpajama
from pm_kvq.quantization.methods.pm_kvq.smoothattention.apply_smoothattention import get_max_keys

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--effective_len", type=int, default=8192)
parser.add_argument("--save_path", type=str)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

calib_dataset = get_calib_redpajama(args.dataset_path, args.n_samples, args.seq_len, tokenizer)
max_keys = get_max_keys(model, calib_dataset, args.effective_len, args.save_path)
