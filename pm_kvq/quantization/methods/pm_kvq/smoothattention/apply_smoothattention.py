from tqdm import tqdm

import torch

from pm_kvq.utils.modeling_utils import get_llm_layers


@torch.no_grad
def get_max_keys(model, calib_dataset, effective_len=None, save_path=None):
    max_keys = []
    if effective_len is not None:
        seq_len = len(calib_dataset[0]["input_ids"])
        scale = effective_len // seq_len
        position_ids = torch.arange(0, effective_len, scale, device=model.device).reshape(1, -1)
    else:
        position_ids = None

    for example in tqdm(calib_dataset, desc="Getting max keys", unit="sample"):
        outputs = model(torch.tensor([example["input_ids"]], device=model.device), position_ids=position_ids, use_cache=True)
        past_keys = outputs.past_key_values.key_cache
        for i, key in enumerate(past_keys):
            key_max = torch.amax(key.abs().reshape(*key.shape[:-2], -1, key.shape[-1] // 2), dim=-2, keepdim=True)
            if len(max_keys) == i:
                max_keys.append(key_max)
            else:
                max_keys[i] = torch.maximum(key_max, max_keys[i])

    if save_path is not None:
        torch.save([max_key.cpu() for max_key in max_keys], save_path)
    return max_keys


def apply_smoothattention_rep(model, max_keys=None, rep_scales=None, alpha=0.5):
    if rep_scales is not None:
        if isinstance(rep_scales, str):
            rep_scales = torch.load(rep_scales)
    elif max_keys is not None:
        if isinstance(max_keys, str):
            max_keys = torch.load(max_keys)
        rep_scales = [max_key**alpha for max_key in max_keys]
    else:
        raise TypeError

    layers = get_llm_layers(model)
    for rep_scale, layer in zip(rep_scales, layers):
        scale = rep_scale.to(layer.self_attn.k_proj.weight.device).clamp(1e-4, 1e4)
        key_scale = scale.repeat(1, 1, 1, 2)
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data / key_scale.reshape(-1, 1)
        if layer.self_attn.k_proj.bias is not None:
            layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data / key_scale.reshape(-1)
        query_scale = torch.repeat_interleave(key_scale, dim=1, repeats=layer.self_attn.num_key_value_groups)
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data * query_scale.reshape(-1, 1)
        if layer.self_attn.q_proj.bias is not None:
            layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data * query_scale.reshape(-1)
