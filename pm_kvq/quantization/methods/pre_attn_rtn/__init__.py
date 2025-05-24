from pm_kvq.quantization.methods.pre_attn_rtn.rtn_modeling_llama import RTN_LlamaAttention_forward
from pm_kvq.quantization.methods.pre_attn_rtn.rtn_modeling_qwen2 import RTN_Qwen2Attention_forward
from pm_kvq.utils.modeling_utils import ModelType, get_model_type

RTN_ATTENTION_FORWARD = {
    ModelType.LLAMA: RTN_LlamaAttention_forward,
    ModelType.QWEN2: RTN_Qwen2Attention_forward,
}
