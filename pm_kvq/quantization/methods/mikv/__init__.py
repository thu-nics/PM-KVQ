from pm_kvq.quantization.methods.mikv.mikv_modeling_llama import MiKV_LlamaAttention_forward
from pm_kvq.quantization.methods.mikv.mikv_modeling_qwen2 import MiKV_Qwen2Attention_forward
from pm_kvq.utils.modeling_utils import ModelType, get_model_type

MIKV_ATTENTION_FORWARD = {
    ModelType.LLAMA: MiKV_LlamaAttention_forward,
    ModelType.QWEN2: MiKV_Qwen2Attention_forward,
}
