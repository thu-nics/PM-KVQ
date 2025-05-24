from pm_kvq.quantization.methods.kivi.kivi_modeling_llama import KIVI_LlamaAttention_forward
from pm_kvq.quantization.methods.kivi.kivi_modeling_qwen2 import KIVI_Qwen2Attention_forward
from pm_kvq.utils.modeling_utils import ModelType, get_model_type

KIVI_ATTENTION_FORWARD = {
    ModelType.LLAMA: KIVI_LlamaAttention_forward,
    ModelType.QWEN2: KIVI_Qwen2Attention_forward,
}
