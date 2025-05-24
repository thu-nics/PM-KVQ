from pm_kvq.quantization.methods.rotatekv.rotatekv_modeling_llama import RotateKV_LlamaAttention_forward
from pm_kvq.quantization.methods.rotatekv.rotatekv_modeling_qwen2 import RotateKV_Qwen2Attention_forward
from pm_kvq.utils.modeling_utils import ModelType, get_model_type

ROTATEKV_ATTENTION_FORWARD = {
    ModelType.LLAMA: RotateKV_LlamaAttention_forward,
    ModelType.QWEN2: RotateKV_Qwen2Attention_forward,
}
