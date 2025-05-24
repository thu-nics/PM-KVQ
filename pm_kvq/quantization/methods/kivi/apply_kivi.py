from types import MethodType

from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
from pm_kvq.quantization.methods.kivi import KIVI_ATTENTION_FORWARD
from pm_kvq.utils.modeling_utils import get_model_type


def apply_kivi(model, k_config, v_config):
    model_type = get_model_type(model)
    k_quantizer = UntrainableQuantizer(**k_config)
    v_quantizer = UntrainableQuantizer(**v_config)
    for layer in model.model.layers:
        layer.self_attn.k_quantizer = k_quantizer
        layer.self_attn.v_quantizer = v_quantizer
        layer.self_attn.kivi_k_quant_len = 0
        layer.self_attn.kivi_v_quant_len = 0
        layer.self_attn.forward = MethodType(KIVI_ATTENTION_FORWARD[model_type], layer.self_attn)
