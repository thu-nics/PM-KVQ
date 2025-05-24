from typing import Callable, Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb, eager_attention_forward, repeat_kv, logger


def MiKV_Qwen2Attention_forward(
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

    ##### MiKV Begin 1/3 #####
    assert input_shape[0] == 1
    if self.n_sink_tokens > 0:
        kwargs["output_attentions"] = True
    if len(past_key_value.key_cache) == self.layer_idx and self.n_sink_tokens > 0:
        self.accumulative_score[:] = 0
        self.sink_token_index = []
    ##### MiKV End 1/3 #####

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    ##### MiKV Begin 2/3 #####
    if input_shape[1] > 1:
        batch, num_key_value_heads, slen, head_dim = key_states.shape
        query_max = torch.amax(query_states.abs().view(batch, num_key_value_heads, self.num_key_value_groups, slen, head_dim), dim=(2, 3)).unsqueeze(-2)
        key_max = torch.amax(key_states.abs(), dim=-2, keepdim=True)
        self.mikv_scales = torch.sqrt(query_max / key_max)
        self.repeat_mikv_scales = repeat_kv(self.mikv_scales, self.num_key_value_groups)
    query_states = query_states / self.repeat_mikv_scales
    key_states = key_states * self.mikv_scales
    ##### MiKV End 2/3 #####

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

    ##### MiKV Begin 3/3 #####
    if past_key_value is not None:
        current_key_value_len = past_key_value.key_cache[self.layer_idx].shape[-2]
        update_key_value_len = input_shape[-1]
        if self.n_sink_tokens > 0:
            accumulative_score = attn_weights.sum(dim=(1, 2))

            if update_key_value_len > 1:
                if update_key_value_len <= self.n_sink_tokens:
                    self.sink_token_index += list(range(current_key_value_len))
                    self.accumulative_score[:, :current_key_value_len] += accumulative_score
                else:
                    sink_token_accumulative_score, sink_token_idx = torch.topk(accumulative_score, self.n_sink_tokens, dim=-1)
                    self.accumulative_score += sink_token_accumulative_score
                    self.sink_token_index += sink_token_idx[-1].tolist()
                    non_sink_token_index = [i for i in range(current_key_value_len) if i not in self.sink_token_index]
                    past_key_value.key_cache[self.layer_idx][..., non_sink_token_index, :] = self.k_quantizer(past_key_value.key_cache[self.layer_idx][..., non_sink_token_index, :])
                    past_key_value.value_cache[self.layer_idx][..., non_sink_token_index, :] = self.v_quantizer(past_key_value.value_cache[self.layer_idx][..., non_sink_token_index, :])
            else:
                if len(self.sink_token_index) < self.n_sink_tokens:
                    self.sink_token_index.append(current_key_value_len - 1)
                    self.accumulative_score[:, :current_key_value_len] += accumulative_score[:, self.sink_token_index]
                else:
                    self.accumulative_score += accumulative_score[:, self.sink_token_index]
                    previous_min, min_idx = torch.min(self.accumulative_score, dim=-1)
                    current_score = accumulative_score[:, current_key_value_len - 1]
                    if previous_min < current_score:
                        past_key_value.key_cache[self.layer_idx][..., self.sink_token_index[min_idx], :] = self.k_quantizer(past_key_value.key_cache[self.layer_idx][..., self.sink_token_index[min_idx], :])
                        past_key_value.value_cache[self.layer_idx][..., self.sink_token_index[min_idx], :] = self.v_quantizer(past_key_value.value_cache[self.layer_idx][..., self.sink_token_index[min_idx], :])
                        self.sink_token_index[min_idx] = current_key_value_len - 1
                        self.accumulative_score[:, min_idx] = current_score
                    else:
                        past_key_value.key_cache[self.layer_idx][..., -1, :] = self.k_quantizer(past_key_value.key_cache[self.layer_idx][..., -1, :])
                        past_key_value.value_cache[self.layer_idx][..., -1, :] = self.v_quantizer(past_key_value.value_cache[self.layer_idx][..., -1, :])
        else:
            past_key_value.key_cache[self.layer_idx][..., -update_key_value_len:, :] = self.k_quantizer(past_key_value.key_cache[self.layer_idx][..., -update_key_value_len:, :])
            past_key_value.value_cache[self.layer_idx][..., -update_key_value_len:, :] = self.v_quantizer(past_key_value.value_cache[self.layer_idx][..., -update_key_value_len:, :])
    ##### MiKV End 3/3 #####

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
