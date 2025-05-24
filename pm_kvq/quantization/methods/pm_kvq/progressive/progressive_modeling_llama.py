from typing import Callable, Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, logger

from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer

quantizers = {
    16: UntrainableQuantizer(n_bits=16, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    8: UntrainableQuantizer(n_bits=8, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    4: UntrainableQuantizer(n_bits=4, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    2: UntrainableQuantizer(n_bits=2, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
    1: UntrainableQuantizer(n_bits=1, granularity="per_group", symmetric=False, group_size=128, round_zeros=False),
}


def Progressive_LlamaAttention_forward(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[DynamicCache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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

    ##### Progressive Begin #####
    if past_key_value is not None:
        current_key_value_len = past_key_value.key_cache[self.layer_idx].shape[-2]
        update_key_value_len = input_shape[-1]
        ori_key_value_len = current_key_value_len - update_key_value_len

        if ori_key_value_len == 0:
            self.n_sink_token_in_memory = 0
            self.n_window_token_in_memory = 0
            self.n_kv_bits = self.n_init_kv_bits
            self.n_bits[:] = -1

        # quantize sink token
        if self.n_sink_token_in_memory < self.n_sink_token:
            n_update_sink_token = min(self.n_sink_token - self.n_sink_token_in_memory, update_key_value_len)
            past_key_value.key_cache[self.layer_idx][..., self.n_sink_token_in_memory : self.n_sink_token_in_memory + n_update_sink_token, :] = quantizers[self.n_sink_token_bits](
                past_key_value.key_cache[self.layer_idx][..., self.n_sink_token_in_memory : self.n_sink_token_in_memory + n_update_sink_token, :]
            )
            past_key_value.value_cache[self.layer_idx][..., self.n_sink_token_in_memory : self.n_sink_token_in_memory + n_update_sink_token, :] = quantizers[self.n_sink_token_bits](
                past_key_value.value_cache[self.layer_idx][..., self.n_sink_token_in_memory : self.n_sink_token_in_memory + n_update_sink_token, :]
            )
            self.n_bits[self.n_sink_token_in_memory : self.n_sink_token_in_memory + n_update_sink_token] = self.n_sink_token_bits
            self.n_sink_token_in_memory += n_update_sink_token
            update_key_value_len -= n_update_sink_token
            ori_key_value_len += n_update_sink_token

        if self.n_window_token > 0:
            if update_key_value_len > 0:
                tmp_window_token_in_memory = self.n_window_token_in_memory + update_key_value_len
                n_out_of_window_token = tmp_window_token_in_memory // self.n_window_token * self.n_window_token
                self.n_window_token_in_memory = tmp_window_token_in_memory % self.n_window_token

                # quantize tokens out of window
                if n_out_of_window_token > 0:
                    past_key_value.key_cache[self.layer_idx][..., -(self.n_window_token_in_memory + n_out_of_window_token) : current_key_value_len - self.n_window_token_in_memory, :] = quantizers[self.n_kv_bits](
                        past_key_value.key_cache[self.layer_idx][..., -(self.n_window_token_in_memory + n_out_of_window_token) : current_key_value_len - self.n_window_token_in_memory, :]
                    )
                    past_key_value.value_cache[self.layer_idx][..., -(self.n_window_token_in_memory + n_out_of_window_token) : current_key_value_len - self.n_window_token_in_memory, :] = quantizers[self.n_kv_bits](
                        past_key_value.value_cache[self.layer_idx][..., -(self.n_window_token_in_memory + n_out_of_window_token) : current_key_value_len - self.n_window_token_in_memory, :]
                    )
                    self.n_bits[current_key_value_len - (self.n_window_token_in_memory + n_out_of_window_token) : current_key_value_len - self.n_window_token_in_memory] = self.n_kv_bits

                    # quantize tokens in window
                    if self.n_window_token_in_memory > 0:
                        past_key_value.key_cache[self.layer_idx][..., -self.n_window_token_in_memory :, :] = quantizers[self.n_window_token_bits](past_key_value.key_cache[self.layer_idx][..., -self.n_window_token_in_memory :, :])
                        past_key_value.value_cache[self.layer_idx][..., -self.n_window_token_in_memory :, :] = quantizers[self.n_window_token_bits](past_key_value.value_cache[self.layer_idx][..., -self.n_window_token_in_memory :, :])
                        self.n_bits[current_key_value_len - self.n_window_token_in_memory : current_key_value_len] = self.n_window_token_bits

                else:
                    # quantize tokens in window
                    past_key_value.key_cache[self.layer_idx][..., -update_key_value_len:, :] = quantizers[self.n_window_token_bits](past_key_value.key_cache[self.layer_idx][..., -update_key_value_len:, :])
                    past_key_value.value_cache[self.layer_idx][..., -update_key_value_len:, :] = quantizers[self.n_window_token_bits](past_key_value.value_cache[self.layer_idx][..., -update_key_value_len:, :])
                    self.n_bits[current_key_value_len - update_key_value_len : current_key_value_len] = self.n_window_token_bits
        else:
            # quantize tokens out of window
            past_key_value.key_cache[self.layer_idx][..., -update_key_value_len:, :] = quantizers[self.n_kv_bits](past_key_value.key_cache[self.layer_idx][..., -update_key_value_len:, :])
            past_key_value.value_cache[self.layer_idx][..., -update_key_value_len:, :] = quantizers[self.n_kv_bits](past_key_value.value_cache[self.layer_idx][..., -update_key_value_len:, :])
            self.n_bits[current_key_value_len - update_key_value_len : current_key_value_len] = self.n_kv_bits

        # check if the kv cache is full
        number_per_token = self.k_proj.out_features + self.v_proj.out_features
        kv_memory = (
            number_per_token
            * (self.n_sink_token_in_memory * self.n_sink_token_bits + self.n_window_token_in_memory * self.n_window_token_bits + max(0, current_key_value_len - self.n_sink_token_in_memory - self.n_window_token_in_memory) * self.n_kv_bits)
            / (8 * 1024 * 1024)
        )
        if kv_memory > self.kv_budget:
            if current_key_value_len <= self.n_sink_token_in_memory + self.n_window_token_in_memory:
                raise MemoryError(f"Memory Error: The kv cache is full, please reduce the kv cache size or increase the kv budget. Current kv budget: {self.kv_budgets}, current kv cache size: {kv_memory}")
            else:
                self.n_kv_bits //= 2
                if self.n_kv_bits < 2:
                    raise MemoryError(f"The kv bits is too small, please increase the kv budget or reduce the kv cache size. Current kv budget: {self.kv_budgets}, current kv cache size: {kv_memory}")
                past_key_value.key_cache[self.layer_idx][..., self.n_sink_token_in_memory : current_key_value_len - self.n_window_token_in_memory, :] = quantizers[self.n_kv_bits](
                    past_key_value.key_cache[self.layer_idx][..., self.n_sink_token_in_memory : current_key_value_len - self.n_window_token_in_memory, :]
                )
                past_key_value.value_cache[self.layer_idx][..., self.n_sink_token_in_memory : current_key_value_len - self.n_window_token_in_memory, :] = quantizers[self.n_kv_bits](
                    past_key_value.value_cache[self.layer_idx][..., self.n_sink_token_in_memory : current_key_value_len - self.n_window_token_in_memory, :]
                )
                self.n_bits[self.n_sink_token_in_memory : current_key_value_len - self.n_window_token_in_memory] = self.n_kv_bits
    ##### Progressive End #####

    return attn_output, attn_weights
