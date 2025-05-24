from typing import Callable, Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, logger

from pm_kvq.utils.hadamard_utils import blockwise_hadamard_transform_cuda


def RotateKV_LlamaAttention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    ##### RotateKV Begin 1/2 #####
    had_dim: Optional[int] = -1,
    k_reorder_indices: Optional[torch.Tensor] = None,
    ##### RotateKV End 1/2 #####
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    ##### RotateKV Begin 2/2 #####
    mode = getattr(self, "mode", None)

    if mode == "calibration":
        key_states = key_states.transpose(1, 2).reshape(input_shape[0] * input_shape[1], -1)
        if not hasattr(self, "channel_mean"):
            self.channel_mean = torch.mean(key_states, dim=0)
        else:
            self.channel_mean += torch.mean(key_states, dim=0)
        assert torch.sum(torch.isinf(self.channel_mean)) == 0

    elif mode == "evaluation":
        # Attention-Sink-Aware Quantization
        for batch_idx in range(key_states.shape[0]):
            quant_mask = torch.full((query_states.shape[2],), True)
            if hasattr(self, "pivot_index_list"):
                quant_mask[self.pivot_index_list[batch_idx, :]] = False
            if input_shape[1] > 1:
                key_states[batch_idx, :, quant_mask, :] = self.k_quantizer(key_states[batch_idx, :, quant_mask, :])
                value_states[batch_idx, :, quant_mask, :] = self.v_quantizer(value_states[batch_idx, :, quant_mask, :])
            else:
                key_states = self.k_quantizer(key_states)
                value_states = self.v_quantizer(value_states)
        if hasattr(self, "pivot_index_list"):
            del self.pivot_index_list

        # Key Reordering
        key_states = key_states.transpose(1, 2).reshape(*input_shape, -1)
        if k_reorder_indices is not None:
            key_states = key_states[..., k_reorder_indices]

    # Grouped-Head Rotation
    dtype = key_states.dtype
    key_states = blockwise_hadamard_transform_cuda(key_states, had_dim, transpose=True).to(dtype)
    key_states = key_states.reshape(hidden_shape).transpose(1, 2)
    ##### RotateKV End 2/2 #####

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
