from types import MethodType

from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
from pm_kvq.quantization.quantizer.shift_quantizer import ShiftQuantizer, ShiftQuantizerWithModification
from pm_kvq.quantization.methods.rtn import RTN_ATTENTION_FORWARD
from pm_kvq.utils.modeling_utils import get_model_type


def apply_rtn(model, k_config=None, v_config=None):
    model_type = get_model_type(model)

    if k_config is not None and v_config is not None:
        k_quantizer = UntrainableQuantizer(**k_config)
        v_quantizer = UntrainableQuantizer(**v_config)
        for layer in model.model.layers:
            layer.self_attn.k_quantizer = k_quantizer
            layer.self_attn.v_quantizer = v_quantizer
            layer.self_attn.forward = MethodType(RTN_ATTENTION_FORWARD[model_type], layer.self_attn)


def apply_shift_rtn(model, k_config=None, v_config=None):
    model_type = get_model_type(model)

    if k_config is not None and v_config is not None:
        k_quantizer = ShiftQuantizer(**k_config)
        v_quantizer = ShiftQuantizer(**v_config)
        for layer in model.model.layers:
            layer.self_attn.k_quantizer = k_quantizer
            layer.self_attn.v_quantizer = v_quantizer
            layer.self_attn.forward = MethodType(RTN_ATTENTION_FORWARD[model_type], layer.self_attn)


def apply_shift_modify_rtn(model, k_config=None, v_config=None):
    model_type = get_model_type(model)

    if k_config is not None and v_config is not None:
        k_quantizer = ShiftQuantizerWithModification(**k_config)
        v_quantizer = ShiftQuantizerWithModification(**v_config)
        for layer in model.model.layers:
            layer.self_attn.k_quantizer = k_quantizer
            layer.self_attn.v_quantizer = v_quantizer
            layer.self_attn.forward = MethodType(RTN_ATTENTION_FORWARD[model_type], layer.self_attn)
