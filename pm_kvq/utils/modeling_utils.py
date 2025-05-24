from enum import Enum, auto


class ModelType(Enum):
    LLAMA = auto()
    QWEN2 = auto()


def get_model_type(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        return ModelType.LLAMA
    elif model.__class__.__name__ == "Qwen2ForCausalLM":
        return ModelType.QWEN2
    else:
        raise NotImplementedError


def get_llm_embedding(model):
    model_type = get_model_type(model)
    if model_type in [ModelType.LLAMA, ModelType.QWEN2]:
        return model.model.embed_tokens
    else:
        raise NotImplementedError


def get_llm_layers(model):
    model_type = get_model_type(model)
    if model_type in [ModelType.LLAMA, ModelType.QWEN2]:
        return model.model.layers
    else:
        raise NotImplementedError


def get_pre_head_layernorm(model):
    model_type = get_model_type(model)
    if model_type in [ModelType.LLAMA, ModelType.QWEN2]:
        return model.model.norm
    else:
        raise NotImplementedError


def get_lm_head(model):
    model_type = get_model_type(model)
    if model_type in [ModelType.LLAMA, ModelType.QWEN2]:
        return model.lm_head
    else:
        raise NotImplementedError


def get_llm_hidden_size(model):
    model_type = get_model_type(model)
    if model_type in [ModelType.LLAMA, ModelType.QWEN2]:
        return model.config.hidden_size
    else:
        raise NotImplementedError


def get_llm_num_attn_heads(model):
    model_type = get_model_type(model)
    if model_type in [ModelType.LLAMA, ModelType.QWEN2]:
        return model.config.num_attention_heads
    else:
        raise NotImplementedError
