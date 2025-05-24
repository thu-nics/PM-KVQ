from types import MethodType

import torch

from pm_kvq.quantization.methods.pm_kvq.progressive import PROGRESSIVE_ATTENTION_FORWARD
from pm_kvq.utils.modeling_utils import get_model_type, get_llm_layers
from pm_kvq.utils.chatbot import _sample


def apply_progressive(
    model,
    kv_budgets,  # in MB
    n_sink_token,
    n_sink_token_bits,
    n_window_token,
    n_window_token_bits,
    n_init_kv_bits,
):
    model_type = get_model_type(model)
    layers = get_llm_layers(model)
    if isinstance(kv_budgets, str):
        kv_budgets = torch.load(kv_budgets)
    if isinstance(kv_budgets, float):
        kv_budgets = [kv_budgets] * len(layers)

    for layer, kv_budget in zip(model.model.layers, kv_budgets):
        layer.self_attn.forward = MethodType(PROGRESSIVE_ATTENTION_FORWARD[model_type], layer.self_attn)
        layer.self_attn.n_sink_token = n_sink_token
        layer.self_attn.n_sink_token_bits = n_sink_token_bits
        layer.self_attn.n_window_token = n_window_token
        layer.self_attn.n_window_token_bits = n_window_token_bits
        layer.self_attn.n_init_kv_bits = n_init_kv_bits
        layer.self_attn.kv_budget = kv_budget
        layer.self_attn.n_bits = torch.ones(35000, dtype=torch.long) * -1

    model._sample = MethodType(_sample, model)
