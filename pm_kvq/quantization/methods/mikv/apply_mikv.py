from types import MethodType

import torch

from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
from pm_kvq.quantization.methods.mikv import MIKV_ATTENTION_FORWARD
from pm_kvq.utils.modeling_utils import get_model_type


def apply_mikv(model, n_sink_tokens, k_config=None, v_config=None):
    model_type = get_model_type(model)

    if k_config is not None and v_config is not None:
        k_quantizer = UntrainableQuantizer(**k_config)
        v_quantizer = UntrainableQuantizer(**v_config)
        for layer in model.model.layers:
            layer.self_attn.n_sink_tokens = n_sink_tokens
            if n_sink_tokens > 0:
                layer.self_attn.accumulative_score = torch.zeros((1, n_sink_tokens), dtype=model.dtype, device=next(layer.parameters()).device)
            layer.self_attn.sink_token_index = []
            layer.self_attn.k_quantizer = k_quantizer
            layer.self_attn.v_quantizer = v_quantizer
            layer.self_attn.forward = MethodType(MIKV_ATTENTION_FORWARD[model_type], layer.self_attn)
