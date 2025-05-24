from tqdm import tqdm
from types import MethodType
from functools import partial

import torch

from pm_kvq.utils.hadamard_utils import apply_hadamard_transform_to_linear
from pm_kvq.utils.modeling_utils import get_model_type, get_llm_layers, get_llm_hidden_size, get_llm_num_attn_heads
from pm_kvq.quantization.methods.rotatekv import ROTATEKV_ATTENTION_FORWARD
from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer


@torch.no_grad
def get_k_reorder_indices(model, calib_dataset, save_path=None, had_dim=-1):
    model_type = get_model_type(model)
    layers = get_llm_layers(model)
    for layer in layers:
        layer.self_attn.mode = "calibration"
        k_proj = layer.self_attn.k_proj
        apply_hadamard_transform_to_linear(k_proj, had_dim=had_dim, output=True)
        layer.self_attn.forward = MethodType(partial(ROTATEKV_ATTENTION_FORWARD[model_type], had_dim=had_dim), layer.self_attn)
    for example in tqdm(calib_dataset, desc="Calibrating", unit="sample"):
        model(torch.tensor([example["input_ids"]], device=model.device), use_cache=False)

    indices = []
    for layer in layers:
        indices.append(torch.argsort(layer.self_attn.channel_mean).cpu())
    if save_path is not None:
        torch.save(indices, save_path)
    return indices


def get_pivot_index_list_hook(module, inputs, n_pivot_token):
    hidden_states = inputs[0]
    if hidden_states.shape[1] > 1:
        pivot_index_list = []
        flat_tensor = torch.flatten(hidden_states, start_dim=1)
        _, topk_indices = torch.topk(torch.abs(flat_tensor), k=n_pivot_token)
        pivot_index_list = topk_indices // hidden_states.shape[-1]
        module.self_attn.pivot_index_list = pivot_index_list


def apply_rotatekv(model, k_config, v_config, k_reorder_indices=None, k_had_dim=-1, n_pivot_token=20):
    model_type = get_model_type(model)
    hidden_size = get_llm_hidden_size(model)
    num_heads = get_llm_num_attn_heads(model)
    head_dim = hidden_size // num_heads
    if isinstance(k_reorder_indices, str):
        k_reorder_indices = torch.load(k_reorder_indices)
    k_quantizer = UntrainableQuantizer(**k_config)
    v_quantizer = UntrainableQuantizer(**v_config)

    layers = get_llm_layers(model)
    for i, layer in enumerate(layers):
        layer.self_attn.mode = "evaluation"
        layer.self_attn.k_quantizer = k_quantizer
        layer.self_attn.v_quantizer = v_quantizer

        # Rotate v_proj and o_proj
        apply_hadamard_transform_to_linear(layer.self_attn.v_proj, had_dim=head_dim, output=True)
        apply_hadamard_transform_to_linear(layer.self_attn.o_proj, had_dim=head_dim, output=False)

        # Rotate k_proj
        apply_hadamard_transform_to_linear(layer.self_attn.k_proj, had_dim=k_had_dim, output=True)

        # Reorder k_proj
        if k_reorder_indices is not None:
            inv_k_reorder_indices = torch.argsort(k_reorder_indices[i])
            layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[k_reorder_indices[i], :]
            if layer.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data[k_reorder_indices[i]]
        else:
            inv_k_reorder_indices = None

        # set pivot token
        if n_pivot_token > 0:
            layer.register_forward_pre_hook(partial(get_pivot_index_list_hook, n_pivot_token=n_pivot_token))

        layer.self_attn.forward = MethodType(partial(ROTATEKV_ATTENTION_FORWARD[model_type], had_dim=k_had_dim, k_reorder_indices=inv_k_reorder_indices), layer.self_attn)
