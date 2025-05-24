import argparse

import torch

from pm_kvq.quantization.methods.pm_kvq.allocation.allocation import allocate_memory_budget

parser = argparse.ArgumentParser()
parser.add_argument("--sensitivity_path", type=str)
parser.add_argument("--memory_budget", type=float)
parser.add_argument("--fbit_choices", type=str)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--max_len", type=int, default=32768)
parser.add_argument("--save_path", type=str, default=None)
args = parser.parse_args()

args.fbit_choices = list(map(int, args.fbit_choices.split(",")))
sensitivity = torch.load(args.sensitivity_path)
allocate_memory_budget(args.fbit_choices, sensitivity["k_sensitivity"], sensitivity["v_sensitivity"], args.memory_budget, args.hidden_size, args.max_len, args.save_path)
