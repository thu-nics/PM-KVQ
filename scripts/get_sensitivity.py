import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pm_kvq.datasets.calib_dataset import get_calib_redpajama
from pm_kvq.quantization.methods.pm_kvq.allocation.sensitivity import get_kv_sensitivity

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--dataset_path", type=str)
parser.add_argument("--n_samples", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--effective_len", type=int, default=8192)
parser.add_argument("--save_path", type=str, default=None)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
hidden_size = model.config.num_key_value_heads * model.config.hidden_size // model.config.num_attention_heads

calib_dataset = get_calib_redpajama(args.dataset_path, args.n_samples, args.seq_len, tokenizer)
k_sensitivity, v_sensitivity = get_kv_sensitivity(model, calib_dataset, args.effective_len, args.save_path)
