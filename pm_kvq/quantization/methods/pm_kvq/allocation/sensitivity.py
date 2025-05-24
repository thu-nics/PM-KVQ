from types import MethodType
from functools import partial
from tqdm import tqdm

import torch

from pm_kvq.utils.modeling_utils import get_model_type, get_llm_layers, ModelType
from pm_kvq.quantization.methods.pm_kvq.allocation.gradkv_modeling_llama import RoPE as LlamaRope, GradKV_LlamaAttention_forward
from pm_kvq.quantization.methods.pm_kvq.allocation.gradkv_modeling_qwen2 import RoPE as Qwen2Rope, GradKV_Qwen2Attention_forward
from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer

quantizers = {
    16: UntrainableQuantizer(n_bits=16, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    8: UntrainableQuantizer(n_bits=8, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    4: UntrainableQuantizer(n_bits=4, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    2: UntrainableQuantizer(n_bits=2, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    1: UntrainableQuantizer(n_bits=1, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
}

ROPE = {
    ModelType.LLAMA: LlamaRope,
    ModelType.QWEN2: Qwen2Rope,
}

GRADKV_FORWARD = {
    ModelType.LLAMA: GradKV_LlamaAttention_forward,
    ModelType.QWEN2: GradKV_Qwen2Attention_forward,
}


def catch_k_hook(rope, inputs, outputs, k_list):
    k_list.append(outputs[1])


def catch_v_hook(v_proj, inputs, outputs, v_list):
    v_list.append(outputs)


def catch_k_grad_hook(rope, grad_output, k_grad_list):
    k_grad_list.append(grad_output[1])


def catch_v_grad_hook(v_proj, grad_output, v_grad_list):
    v_grad_list.append(grad_output[0])


def get_kv_sensitivity(model, calib_dataset, effective_len=None, save_path=None):
    model_type = get_model_type(model)
    layers = get_llm_layers(model)
    n_layers = len(layers)
    n_samples = len(calib_dataset)
    rope_type = ROPE[model_type]

    if effective_len is not None:
        seq_len = len(calib_dataset[0]["input_ids"])
        scale = effective_len // seq_len
        position_ids = torch.arange(0, effective_len, scale, device=model.device).reshape(1, -1)
    else:
        position_ids = None

    k_list = []
    v_list = []
    k_grad_list = []
    v_grad_list = []
    k_sensitivity = {16: [0] * n_layers, 8: [0] * n_layers, 4: [0] * n_layers, 2: [0] * n_layers, 1: [0] * n_layers}
    v_sensitivity = {16: [0] * n_layers, 8: [0] * n_layers, 4: [0] * n_layers, 2: [0] * n_layers, 1: [0] * n_layers}

    for layer in layers:
        layer.self_attn.rope = rope_type()
        layer.self_attn.forward = MethodType(GRADKV_FORWARD[model_type], layer.self_attn)
        layer.self_attn.rope.register_full_backward_pre_hook(partial(catch_k_grad_hook, k_grad_list=k_grad_list))
        layer.self_attn.rope.register_forward_hook(partial(catch_k_hook, k_list=k_list))
        layer.self_attn.v_proj.register_full_backward_pre_hook(partial(catch_v_grad_hook, v_grad_list=v_grad_list))
        layer.self_attn.v_proj.register_forward_hook(partial(catch_v_hook, v_list=v_list))

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    layers[0].input_layernorm.weight.requires_grad = True

    for example in tqdm(calib_dataset, desc="Profiling sensitivity", unit="sample"):
        loss = model(torch.tensor([example["input_ids"]], device=model.device), labels=torch.tensor([example["input_ids"]], device=model.device), position_ids=position_ids, use_cache=False).loss
        loss.backward()

        with torch.no_grad():
            for i, (key_states, value_states, key_grad, value_grad) in enumerate(zip(k_list, v_list, k_grad_list, v_grad_list)):
                for n_bits in [16, 8, 4, 2, 1]:
                    k_sensitivity[n_bits][i] += ((quantizers[n_bits](key_states) - key_states) * key_grad.to(key_states.device)).abs().mean().item()
                    v_sensitivity[n_bits][i] += ((quantizers[n_bits](value_states) - value_states) * value_grad.to(value_states.device)).abs().mean().item()

        k_list.clear()
        v_list.clear()
        k_grad_list.clear()
        v_grad_list.clear()

    for i in range(n_layers):
        for n_bits in [16, 8, 4, 2, 1]:
            k_sensitivity[n_bits][i] /= n_samples
            v_sensitivity[n_bits][i] /= n_samples
    if save_path is not None:
        torch.save({"k_sensitivity": k_sensitivity, "v_sensitivity": v_sensitivity}, save_path)

    return k_sensitivity, v_sensitivity
