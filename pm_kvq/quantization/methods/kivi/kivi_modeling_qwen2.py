from typing import Callable, Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, eager_attention_forward, logger


def KIVI_Qwen2Attention_forward(
    self: Qwen2Attention,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    ##### KIVI Begin 1/3 #####
    past_key_value_len = past_key_value.key_cache[self.layer_idx].shape[-2] if past_key_value is not None and len(past_key_value.key_cache) > self.layer_idx else 0
    if past_key_value_len == 0:
        self.kivi_k_quant_len = 0
        self.kivi_v_quant_len = 0
    ##### KIVI End 1/3 #####

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    ##### KIVI Begin 2/3 #####
    if input_shape[1] == 1 and (past_key_value_len + 1) % self.k_quantizer.group_size == 0:
        key_states[..., -self.k_quantizer.group_size :, :] = self.k_quantizer(key_states[..., -self.k_quantizer.group_size :, :].transpose(-1, -2)).transpose(-1, -2)
        self.kivi_k_quant_len += self.k_quantizer.group_size
    if input_shape[1] == 1 and (past_key_value_len + 1) % self.v_quantizer.group_size == 0:
        value_states[..., -self.v_quantizer.group_size :, :] = self.v_quantizer(value_states[..., -self.v_quantizer.group_size :, :])
        self.kivi_v_quant_len += self.v_quantizer.group_size
    ##### KIVI End 2/3 #####

    sliding_window = None
    if self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None and self.layer_idx >= self.config.max_window_layers:
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to " 'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    ##### KIVI Begin 3/3 #####
    if input_shape[1] > 1:
        kivi_k_update_quant_len = (past_key_value_len + input_shape[1]) // self.k_quantizer.group_size * self.k_quantizer.group_size
        kivi_v_update_quant_len = (past_key_value_len + input_shape[1]) // self.v_quantizer.group_size * self.v_quantizer.group_size
        if kivi_k_update_quant_len > self.kivi_k_quant_len:
            past_key_value.key_cache[self.layer_idx][..., self.kivi_k_quant_len : kivi_k_update_quant_len, :] = self.k_quantizer(
                past_key_value.key_cache[self.layer_idx][..., self.kivi_k_quant_len : kivi_k_update_quant_len, :].transpose(-1, -2),
            ).transpose(-1, -2)
        if kivi_v_update_quant_len > self.kivi_v_quant_len:
            past_key_value.value_cache[self.layer_idx][..., self.kivi_v_quant_len : kivi_v_update_quant_len, :] = self.v_quantizer(
                past_key_value.value_cache[self.layer_idx][..., self.kivi_v_quant_len : kivi_v_update_quant_len, :],
            )
        self.kivi_k_quant_len = kivi_k_update_quant_len
        self.kivi_v_quant_len = kivi_v_update_quant_len
    ##### KIVI End 3/3 #####

    return attn_output, attn_weights
