from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
from pm_kvq.quantization.methods.pm_kvq.progressive.progressive_modeling_qwen2 import Progressive_Qwen2Attention_forward
from pm_kvq.quantization.methods.pm_kvq.progressive.progressive_modeling_llama import Progressive_LlamaAttention_forward
from pm_kvq.utils.modeling_utils import ModelType, get_model_type

PROGRESSIVE_ATTENTION_FORWARD = {
    ModelType.QWEN2: Progressive_Qwen2Attention_forward,
    ModelType.LLAMA: Progressive_LlamaAttention_forward,
}
