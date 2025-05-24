# PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

## Abstract

Recently, significant progress has been made in developing reasoning-capable Large Language Models (LLMs) through long Chain-of-Thought (CoT) techniques. However, this long-CoT reasoning process imposes substantial memory overhead due to the large Key-Value (KV) Cache memory overhead. Post-training KV Cache quantization has emerged as a promising compression technique and has been extensively studied in short-context scenarios. However, directly applying existing methods to long-CoT LLMs causes significant performance degradation due to the following two reasons: (1) **Large cumulative error**: Existing methods fail to adequately leverage available memory, and they directly quantize the KV Cache during each decoding step, leading to large cumulative quantization error. (2) **Short-context calibration**: Due to Rotary Positional Embedding (RoPE), the use of short-context data during calibration fails to account for the distribution of less frequent channels in the Key Cache, resulting in performance loss. We propose **P**rogressive **M**ixed-Precision **KV** Cache **Q**uantization (**PM-KVQ**) for long-CoT LLMs to address the above issues in two folds: (1) To reduce cumulative error, we design a progressive quantization strategy to gradually lower the bit-width of KV Cache in each block. Then, we propose block-wise memory allocation to assign a higher bit-width to more sensitive transformer blocks. (2) To increase the calibration length without additional overhead, we propose a new calibration strategy with positional interpolation that leverages short calibration data with positional interpolation to approximate the data distribution of long-context data. Extensive experiments on 7B–70B long-CoT LLMs show that PM-KVQ improves reasoning benchmark performance by up to 8% over SOTA baselines under the same memory budget.

## Installation

1. Create a new conda environment.

   ```bash
   conda create -n pm_kvq python==3.10
   conda activate pm_kvq
   ```

2. Use pip to install packages from requirements.

   ```bash
   pip install -r requirements.txt
   ```

3. Install `pm_kvq` from source.

   ```bash
   pip install -e .
   ```

4. For RotateKV baseline, install `fast-hadamard-transform` from [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform).

## Apply PM-KVQ

### Block-wise Memory Allocation

1. Profile the sensitivity to quantization of KV Cache in different transformer blocks.

   ```bash
   python scripts/get_sensitivity.py \
   --model_path /PATH/TO/MODEL \
   --dataset_path /PATH/TO/CALIBRATION/DATASET \
   --n_samples 512 \
   --seq_len 2048 \
   --effective_len 8192 \
   --save_path /PATH/TO/SAVE/SENSITIVITY
   ```

2. Assign memory budget to each transformer block. The value of `--memory_budget` is specified in megabytes (MB).

   ```bash
   python scripts/allocate_memory.py \
   --sensitivity_path /PATH/TO/SENSITIVITY \
   --memory_budget 1024 \
   --fbit_choices 4,2 \
   --hidden_size ${HIDDEN_DIMENSION_OF_MODEL} \
   --max_len 32768 \
   --save_path /PATH/TO/SAVE/MEMORY/BUDGET
   ```

### Calibration with Positional Interpolation

1. Calculate maximum magnitude of the Key cache.

   ```bash
   python scripts/get_max_keys.py \
   --model_path /PATH/TO/MODEL \
   --dataset_path /PATH/TO/CALIBRATION/DATASET \
   --n_samples 512 \
   --seq_len 2048 \
   --effective_len 8192 \
   --save_path /PATH/TO/SAVE/MAX/KEYS
   ```

2. Search for the optimal reparameterization factor.

   ```bash
   python scripts/search_rep_scales.py \
   --model_path /PATH/TO/MODEL \
   --dataset_path /PATH/TO/CALIBRATION/DATASET \
   --n_samples 512 \
   --seq_len 2048 \
   --effective_len 8192 \
   --max_keys_path /PATH/TO/MAX/KEYS \
   --k_bits 4 \
   --v_bits 4 \
   --save_path /PATH/TO/SAVE/REP/SCALES
   ```

### Quantization and Evaluation

1. Evaluate the quantized model and save its responses to a `.jsonl` file. Use the `--start` and `--end` options to specify the range of problem indices to evaluate. To facilitate joint judgement, save the response files for different problems in the same directory.

   ```bash
   python scripts/evaluation.py \
   --model_path /PATH/TO/MODEL \
   --output_path /PATH/TO/SAVE/MODEL/RESPONSES \
   --benchmark aime \
   --version 2024 \
   --start 0 \
   --end 30 \
   --n_responses 16 \
   --method pm-kvq \
   --backend fake \
   --rep_scales /PATH/TO/REP/SCALES \
   --kv_budgets /PATH/TO/MEMORY/BUDGET \
   --n_sink_token 1 \
   --n_sink_token_bits 16 \
   --n_window_token 128 \
   --n_window_token_bits 16 \
   --n_init_kv_bits 16
   ```

2. Judge the responses and calculate the evaluation metrics.

   ```bash
   python scripts/judge.py \
   --benchmark aime \
   --version 2024 \
   --responses_dir /PATH/TO/MODEL/RESPONSES
   ```

## Contact us

- Tengxuan Liu: [liutx21@mails.tsinghua.edu.cn](mailto:liutx21@mails.tsinghua.edu.cn)
- Shiyao Li: [lishiyao20@mails.tsinghua.edu.cn](mailto:lishiyao20@mails.tsinghua.edu.cn)
- Jiayi Yang: [jy.yang1030@gmail.com](mailto:jy.yang1030@gmail.com)
- Yu Wang: [yu-wang@tsinghua.edu.cn](mailto:yu-wang@tsinghua.edu.cn)

This work is maintained by [NICS-EFC Lab](https://nicsefc.ee.tsinghua.edu.cn/) (Tsinghua University) and [Infinigence-AI](https://www.infini-ai.com/) (Beijing China).



<p align="middle">
  <img src="figures/logo_nicsefc.jpg" width="35%" hspace="30" />
  <img src="figures/logo_Infinigence-ai.png" width="35%" hspace="30" />
</p>