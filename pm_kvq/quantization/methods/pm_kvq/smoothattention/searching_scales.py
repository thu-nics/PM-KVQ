from copy import deepcopy
from functools import partial
from types import MethodType
from tqdm import tqdm

import torch

from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
from pm_kvq.quantization.methods.pre_attn_rtn import RTN_ATTENTION_FORWARD
from pm_kvq.utils.modeling_utils import get_llm_layers, get_llm_hidden_size, get_model_type


def catch_feature_hook(module, args, kwargs, hidden_states, position_embeddings):
    n = hidden_states["batch"]
    hidden_states["hidden_states"][n : n + 1] = args[0].to(hidden_states["hidden_states"].device) if len(args) > 0 else kwargs["hidden_states"].to(hidden_states["hidden_states"].device)
    if len(position_embeddings) == 0:
        position_embeddings.append(kwargs["position_embeddings"][0].cuda())
        position_embeddings.append(kwargs["position_embeddings"][1].cuda())
    raise ValueError


def catch_attention_ouptut_hook(module, inputs, outputs, attention_outputs, is_teacher, batch_size):
    if is_teacher:
        n = attention_outputs["batch"]
        attention_outputs["original"][n * batch_size : (n + 1) * batch_size] = outputs[0]
    else:
        attention_outputs["quant"] = outputs[0]
    raise ValueError


@torch.no_grad
def search_rep_scales(model, k_config, v_config, dataset, max_keys, grid=20, batch_size=1, effective_len=None, save_path=None):
    model_type = get_model_type(model)
    layers = get_llm_layers(model)
    k_quantizer = UntrainableQuantizer(**k_config)
    v_quantizer = UntrainableQuantizer(**v_config)
    n_samples = len(dataset)
    seq_len = len(dataset[0]["input_ids"])
    hidden_size = get_llm_hidden_size(model)

    hidden_states = {"hidden_states": torch.zeros((n_samples, seq_len, hidden_size), device="cuda", dtype=model.dtype), "batch": 0}
    position_embeddings = []
    if effective_len is not None:
        seq_len = len(dataset[0]["input_ids"])
        scale = effective_len // seq_len
        position_ids = torch.arange(0, effective_len, scale, device=model.device).reshape(1, -1)
    else:
        position_ids = None
    handle = layers[0].register_forward_pre_hook(partial(catch_feature_hook, hidden_states=hidden_states, position_embeddings=position_embeddings), with_kwargs=True)
    with torch.no_grad():
        for example in tqdm(dataset, desc="Generating Features"):
            try:
                model(torch.tensor([example["input_ids"]], device=model.device), position_ids=position_ids)
            except ValueError:
                hidden_states["batch"] += 1
                continue
    handle.remove()

    input_hidden_states = hidden_states["hidden_states"]
    attention_outputs = {
        "original": torch.zeros_like(hidden_states["hidden_states"]),
        "quant": None,
    }
    result_rep_scales = []
    result_alpha = []
    loss_func = torch.nn.MSELoss()

    for n_layer, layer in enumerate(tqdm(layers)):
        layer.cuda()
        quant_layer = deepcopy(layer)
        handle = layer.self_attn.register_forward_hook(partial(catch_attention_ouptut_hook, attention_outputs=attention_outputs, is_teacher=True, batch_size=batch_size))
        attention_outputs["batch"] = 0
        for i in range(n_samples // batch_size):
            try:
                layer(input_hidden_states[i * batch_size : (i + 1) * batch_size], position_embeddings=position_embeddings)[0]
            except ValueError:
                attention_outputs["batch"] += 1
                pass
        handle.remove()

        quant_layer.self_attn.k_quantizer = k_quantizer
        quant_layer.self_attn.v_quantizer = v_quantizer
        quant_layer.self_attn.forward = MethodType(RTN_ATTENTION_FORWARD[model_type], quant_layer.self_attn)

        quant_handle = quant_layer.self_attn.register_forward_hook(partial(catch_attention_ouptut_hook, attention_outputs=attention_outputs, is_teacher=False, batch_size=batch_size))
        min_loss = float("inf")
        loss_all = []
        best_rep_scales = None
        best_alpha = None

        for n in range(grid):
            loss_all.append(0)
            rep_scales = (max_keys[n_layer].to(next(layer.parameters()).device, next(layer.parameters()).dtype) ** (1 - n / grid)).clamp(min=1e-4, max=1e4)
            key_scales = rep_scales.repeat(1, 1, 1, 2)
            quant_layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data / key_scales.reshape(-1, 1)
            if quant_layer.self_attn.k_proj.bias is not None:
                quant_layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data / key_scales.reshape(-1)
            query_scale = torch.repeat_interleave(key_scales, dim=1, repeats=layer.self_attn.num_key_value_groups)
            quant_layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data * query_scale.reshape(-1, 1)
            if quant_layer.self_attn.q_proj.bias is not None:
                quant_layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data * query_scale.reshape(-1)

            for i in range(n_samples // batch_size):
                try:
                    quant_layer(input_hidden_states[i * batch_size : (i + 1) * batch_size], position_embeddings=position_embeddings)[0]
                except ValueError:
                    pass
                loss = loss_func(attention_outputs["original"][i * batch_size : (i + 1) * batch_size].float(), attention_outputs["quant"].float()).item()
                loss_all[-1] += loss

            if loss_all[-1] < min_loss:
                min_loss = loss_all[-1]
                best_rep_scales = rep_scales
                best_alpha = 1 - n / grid

        assert best_rep_scales is not None
        result_rep_scales.append(best_rep_scales.cpu())
        result_alpha.append(best_alpha)
        quant_handle.remove()

        for i in range(n_samples // batch_size):
            input_hidden_states[i * batch_size : (i + 1) * batch_size] = layer(input_hidden_states[i * batch_size : (i + 1) * batch_size], position_embeddings=position_embeddings)[0]

        layer.cpu()
        del quant_layer
        torch.cuda.empty_cache()

    if save_path is not None:
        torch.save(result_rep_scales, save_path)
    return result_rep_scales
