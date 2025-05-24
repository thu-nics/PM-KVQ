from typing import List, Optional

import torch
from transformers.cache_utils import Cache, CacheConfig

from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer
from pm_kvq.quantization.quantizer.pack_funcs import unpack_funcs, pack_funcs


class ProgressiveQuantizedCacheConfig(CacheConfig):

    def __init__(
        self,
        kv_budgets,
        n_sink_token=1,
        n_sink_token_bits=16,
        n_window_token=128,
        n_window_token_bits=16,
        n_init_kv_bits=16,
    ):
        self.kv_budgets = kv_budgets
        self.n_sink_token = n_sink_token
        self.n_sink_token_bits = n_sink_token_bits
        self.n_window_token = n_window_token
        self.n_window_token_bits = n_window_token_bits
        self.n_init_kv_bits = n_init_kv_bits


class ProgressiveQuantizedCache(Cache):
    def __init__(self, cache_config: ProgressiveQuantizedCacheConfig):
        super().__init__()

        self.kv_budgets = cache_config.kv_budgets
        self.n_sink_token = cache_config.n_sink_token
        self.n_sink_token_bits = cache_config.n_sink_token_bits
        self.n_window_token = cache_config.n_window_token
        self.n_window_token_bits = cache_config.n_window_token_bits
        self.n_init_kv_bits = cache_config.n_init_kv_bits
        self.n_kv_bits = []
        self.quantizers = {
            16: UntrainableQuantizer(n_bits=16, granularity="per_group", symmetric=False, group_size=128, round_zeros=False, pack=True),
            8: UntrainableQuantizer(n_bits=8, granularity="per_group", symmetric=False, group_size=128, round_zeros=False, pack=True),
            4: UntrainableQuantizer(n_bits=4, granularity="per_group", symmetric=False, group_size=128, round_zeros=False, pack=True),
            2: UntrainableQuantizer(n_bits=2, granularity="per_group", symmetric=False, group_size=128, round_zeros=False, pack=True),
            1: UntrainableQuantizer(n_bits=1, granularity="per_group", symmetric=False, group_size=128, round_zeros=False, pack=True),
        }

        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.key_scales: List[torch.Tensor] = []
        self.key_zeros: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.value_scales: List[torch.Tensor] = []
        self.value_zeros: List[torch.Tensor] = []

        if self.n_sink_token > 0:
            self.sink_token_key_cache: List[torch.Tensor] = []
            self.sink_token_key_scales: List[torch.Tensor] = []
            self.sink_token_key_zeros: List[torch.Tensor] = []
            self.sink_token_value_cache: List[torch.Tensor] = []
            self.sink_token_value_scales: List[torch.Tensor] = []
            self.sink_token_value_zeros: List[torch.Tensor] = []

        if self.n_window_token > 0:
            self.window_token_key_cache: List[torch.Tensor] = []
            self.window_token_key_scales: List[torch.Tensor] = []
            self.window_token_key_zeros: List[torch.Tensor] = []
            self.window_token_value_cache: List[torch.Tensor] = []
            self.window_token_value_scales: List[torch.Tensor] = []
            self.window_token_value_zeros: List[torch.Tensor] = []

        self.n_sink_token_in_memory = []
        self.n_window_token_in_memory = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) == layer_idx:
            keys_to_return = key_states
            values_to_return = value_states
        else:
            past_keys, past_values = self.dequantize(layer_idx, device=key_states.device)
            keys_to_return = torch.cat([past_keys, key_states], dim=-2)
            values_to_return = torch.cat([past_values, value_states], dim=-2)
        self.quantize(key_states, value_states, layer_idx, cache_kwargs)

        return keys_to_return, values_to_return

    def quantize(self, key_states, value_states, layer_idx, cache_kwargs=None):
        update_key_value_len = key_states.shape[-2]
        if len(self.n_sink_token_in_memory) == layer_idx:
            self.n_sink_token_in_memory.append(0)
            self.sink_token_key_cache.append(torch.zeros(0, dtype=torch.int16, device=key_states.device))
            self.sink_token_key_scales.append(torch.zeros(0, dtype=torch.bfloat16, device=key_states.device))
            self.sink_token_key_zeros.append(torch.zeros(0, dtype=torch.bfloat16, device=key_states.device))
            self.sink_token_value_cache.append(torch.zeros(0, dtype=torch.int16, device=value_states.device))
            self.sink_token_value_scales.append(torch.zeros(0, dtype=torch.bfloat16, device=value_states.device))
            self.sink_token_value_zeros.append(torch.zeros(0, dtype=torch.bfloat16, device=value_states.device))

        if len(self.n_window_token_in_memory) == layer_idx:
            self.n_window_token_in_memory.append(0)
            self.window_token_key_cache.append(torch.zeros(0, dtype=torch.int16, device=key_states.device))
            self.window_token_key_scales.append(torch.zeros(0, dtype=torch.bfloat16, device=key_states.device))
            self.window_token_key_zeros.append(torch.zeros(0, dtype=torch.bfloat16, device=key_states.device))
            self.window_token_value_cache.append(torch.zeros(0, dtype=torch.int16, device=value_states.device))
            self.window_token_value_scales.append(torch.zeros(0, dtype=torch.bfloat16, device=value_states.device))
            self.window_token_value_zeros.append(torch.zeros(0, dtype=torch.bfloat16, device=value_states.device))

        if len(self.key_cache) == layer_idx:
            self.key_cache.append(torch.zeros(0, dtype=torch.int16, device=key_states.device))
            self.key_scales.append(torch.zeros(0, dtype=torch.bfloat16, device=key_states.device))
            self.key_zeros.append(torch.zeros(0, dtype=torch.bfloat16, device=key_states.device))
            self.value_cache.append(torch.zeros(0, dtype=torch.int16, device=key_states.device))
            self.value_scales.append(torch.zeros(0, dtype=torch.bfloat16, device=value_states.device))
            self.value_zeros.append(torch.zeros(0, dtype=torch.bfloat16, device=value_states.device))

        if len(self.n_kv_bits) == layer_idx:
            self.n_kv_bits.append(self.n_init_kv_bits)

        if self.n_sink_token_in_memory[layer_idx] < self.n_sink_token:
            n_update_sink_token = min(self.n_sink_token - self.n_sink_token_in_memory[layer_idx], update_key_value_len)

            quantized_sink_token_key, sink_token_key_scales, sink_token_key_zeros = self.quantizers[self.n_sink_token_bits].real_quant(key_states[..., :n_update_sink_token, :], return_params=True)
            self.sink_token_key_cache[layer_idx] = torch.cat([self.sink_token_key_cache[layer_idx], quantized_sink_token_key], dim=-2)
            self.sink_token_key_scales[layer_idx] = torch.cat([self.sink_token_key_scales[layer_idx], sink_token_key_scales], dim=-2)
            self.sink_token_key_zeros[layer_idx] = torch.cat([self.sink_token_key_zeros[layer_idx], sink_token_key_zeros], dim=-2)

            quantized_sink_token_value, sink_token_value_scales, sink_token_value_zeros = self.quantizers[self.n_sink_token_bits].real_quant(value_states[..., :n_update_sink_token, :], return_params=True)
            self.sink_token_value_cache[layer_idx] = torch.cat([self.sink_token_value_cache[layer_idx], quantized_sink_token_value], dim=-2)
            self.sink_token_value_scales[layer_idx] = torch.cat([self.sink_token_value_scales[layer_idx], sink_token_value_scales], dim=-2)
            self.sink_token_value_zeros[layer_idx] = torch.cat([self.sink_token_value_zeros[layer_idx], sink_token_value_zeros], dim=-2)

            self.n_sink_token_in_memory[layer_idx] += n_update_sink_token
            update_key_value_len -= n_update_sink_token
        else:
            n_update_sink_token = 0

        if self.n_window_token > 0:
            if update_key_value_len > 0:
                tmp_window_token_in_memory = self.n_window_token_in_memory[layer_idx] + update_key_value_len
                n_out_of_window_token = tmp_window_token_in_memory // self.n_window_token * self.n_window_token
                self.n_window_token_in_memory[layer_idx] = tmp_window_token_in_memory % self.n_window_token

            if n_out_of_window_token > 0:
                quantized_window_token_key, window_token_key_scales, window_token_key_zeros = self.quantizers[self.n_window_token_bits].real_quant(key_states[..., n_update_sink_token : n_update_sink_token + n_out_of_window_token, :], return_params=True)
                self.window_token_key_cache[layer_idx] = torch.cat([self.window_token_key_cache[layer_idx], quantized_window_token_key], dim=-2)
                self.window_token_key_scales[layer_idx] = torch.cat([self.window_token_key_scales[layer_idx], window_token_key_scales], dim=-2)
                self.window_token_key_zeros[layer_idx] = torch.cat([self.window_token_key_zeros[layer_idx], window_token_key_zeros], dim=-2)
                quantized_window_token_value, window_token_value_scales, window_token_value_zeros = self.quantizers[self.n_window_token_bits].real_quant(
                    value_states[..., n_update_sink_token : n_update_sink_token + n_out_of_window_token, :], return_params=True
                )
                self.window_token_value_cache[layer_idx] = torch.cat([self.window_token_value_cache[layer_idx], quantized_window_token_value], dim=-2)
                self.window_token_value_scales[layer_idx] = torch.cat([self.window_token_value_scales[layer_idx], window_token_value_scales], dim=-2)
                self.window_token_value_zeros[layer_idx] = torch.cat([self.window_token_value_zeros[layer_idx], window_token_value_zeros], dim=-2)

                n_tmp_window_token_bits = self.n_window_token_bits
                while n_tmp_window_token_bits > self.n_kv_bits[layer_idx]:
                    self.window_token_key_cache[layer_idx], self.window_token_key_scales[layer_idx], self.window_token_key_zeros[layer_idx] = self.bit_width_shrinking(
                        self.window_token_key_cache[layer_idx], self.window_token_key_scales[layer_idx], self.window_token_key_zeros[layer_idx], n_tmp_window_token_bits
                    )
                    self.window_token_value_cache[layer_idx], self.window_token_value_scales[layer_idx], self.window_token_value_zeros[layer_idx] = self.bit_width_shrinking(
                        self.window_token_value_cache[layer_idx], self.window_token_value_scales[layer_idx], self.window_token_value_zeros[layer_idx], n_tmp_window_token_bits
                    )
                    n_tmp_window_token_bits //= 2
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], self.window_token_key_cache[layer_idx]], dim=-2)
                self.key_scales[layer_idx] = torch.cat([self.key_scales[layer_idx], self.window_token_key_scales[layer_idx]], dim=-2)
                self.key_zeros[layer_idx] = torch.cat([self.key_zeros[layer_idx], self.window_token_key_zeros[layer_idx]], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], self.window_token_value_cache[layer_idx]], dim=-2)
                self.value_scales[layer_idx] = torch.cat([self.value_scales[layer_idx], self.window_token_value_scales[layer_idx]], dim=-2)
                self.value_zeros[layer_idx] = torch.cat([self.value_zeros[layer_idx], self.window_token_value_zeros[layer_idx]], dim=-2)

                self.window_token_key_cache[layer_idx] = torch.zeros(0, dtype=torch.int16, device=key_states.device)
                self.window_token_key_scales[layer_idx] = torch.zeros(0, dtype=torch.bfloat16, device=key_states.device)
                self.window_token_key_zeros[layer_idx] = torch.zeros(0, dtype=torch.bfloat16, device=key_states.device)
                self.window_token_value_cache[layer_idx] = torch.zeros(0, dtype=torch.int16, device=key_states.device)
                self.window_token_value_scales[layer_idx] = torch.zeros(0, dtype=torch.bfloat16, device=value_states.device)
                self.window_token_value_zeros[layer_idx] = torch.zeros(0, dtype=torch.bfloat16, device=value_states.device)

                if self.n_window_token_in_memory[layer_idx] > 0:
                    quantized_window_token_key, window_token_key_scales, window_token_key_zeros = self.quantizers[self.n_window_token_bits].real_quant(key_states[..., -self.n_window_token_in_memory[layer_idx] :, :], return_params=True)
                    self.window_token_key_cache[layer_idx] = torch.cat([self.window_token_key_cache[layer_idx], quantized_window_token_key], dim=-2)
                    self.window_token_key_scales[layer_idx] = torch.cat([self.window_token_key_scales[layer_idx], window_token_key_scales], dim=-2)
                    self.window_token_key_zeros[layer_idx] = torch.cat([self.window_token_key_zeros[layer_idx], window_token_key_zeros], dim=-2)
                    quantized_window_token_value, window_token_value_scales, window_token_value_zeros = self.quantizers[self.n_window_token_bits].real_quant(value_states[..., -self.n_window_token_in_memory[layer_idx] :, :], return_params=True)
                    self.window_token_value_cache[layer_idx] = torch.cat([self.window_token_value_cache[layer_idx], quantized_window_token_value], dim=-2)
                    self.window_token_value_scales[layer_idx] = torch.cat([self.window_token_value_scales[layer_idx], window_token_value_scales], dim=-2)
                    self.window_token_value_zeros[layer_idx] = torch.cat([self.window_token_value_zeros[layer_idx], window_token_value_zeros], dim=-2)

            else:
                quantized_window_token_key, window_token_key_scales, window_token_key_zeros = self.quantizers[self.n_window_token_bits].real_quant(key_states[..., n_update_sink_token:, :], return_params=True)
                self.window_token_key_cache[layer_idx] = torch.cat([self.window_token_key_cache[layer_idx], quantized_window_token_key], dim=-2)
                self.window_token_key_scales[layer_idx] = torch.cat([self.window_token_key_scales[layer_idx], window_token_key_scales], dim=-2)
                self.window_token_key_zeros[layer_idx] = torch.cat([self.window_token_key_zeros[layer_idx], window_token_key_zeros], dim=-2)
                quantized_window_token_value, window_token_value_scales, window_token_value_zeros = self.quantizers[self.n_window_token_bits].real_quant(value_states[..., n_update_sink_token:, :], return_params=True)
                self.window_token_value_cache[layer_idx] = torch.cat([self.window_token_value_cache[layer_idx], quantized_window_token_value], dim=-2)
                self.window_token_value_scales[layer_idx] = torch.cat([self.window_token_value_scales[layer_idx], window_token_value_scales], dim=-2)
                self.window_token_value_zeros[layer_idx] = torch.cat([self.window_token_value_zeros[layer_idx], window_token_value_zeros], dim=-2)

        else:
            quantized_key, key_scales, key_zeros = self.quantizers[self.n_window_token_bits].real_quant(key_states[..., n_update_sink_token:, :], return_params=True)
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], quantized_key], dim=-2)
            self.key_scales[layer_idx] = torch.cat([self.key_scales[layer_idx], key_scales], dim=-2)
            self.key_zeros[layer_idx] = torch.cat([self.key_zeros[layer_idx], key_zeros], dim=-2)
            quantized_value, value_scales, value_zeros = self.quantizers[self.n_window_token_bits].real_quant(value_states[..., n_update_sink_token:, :], return_params=True)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], quantized_value], dim=-2)
            self.value_scales[layer_idx] = torch.cat([self.value_scales[layer_idx], value_scales], dim=-2)
            self.value_zeros[layer_idx] = torch.cat([self.value_zeros[layer_idx], value_zeros], dim=-2)

        kv_memory = (
            (
                self.key_cache[layer_idx].numel()
                + self.value_cache[layer_idx].numel()
                + self.sink_token_key_cache[layer_idx].numel()
                + self.sink_token_value_cache[layer_idx].numel()
                + self.window_token_key_cache[layer_idx].numel()
                + self.window_token_value_cache[layer_idx].numel()
            )
            * 2
            / (1024 * 1024)
        )

        if kv_memory > self.kv_budgets[layer_idx]:
            if self.n_sink_token_in_memory[layer_idx] + self.n_window_token_in_memory[layer_idx] < self.n_sink_token + self.n_window_token and self.key_cache[layer_idx].shape[-2] == 0:
                raise MemoryError(f"Memory Error: The kv cache is full, please reduce the kv cache size or increase the kv budget. Current kv budget: {self.kv_budgets}, current kv cache size: {kv_memory}")
            else:
                if self.n_kv_bits[layer_idx] <= 2:
                    raise MemoryError(f"The kv bits is too small, please increase the kv budget or reduce the kv cache size. Current kv budget: {self.kv_budgets}, current kv cache size: {kv_memory}")
                self.key_cache[layer_idx], self.key_scales[layer_idx], self.key_zeros[layer_idx] = self.bit_width_shrinking(self.key_cache[layer_idx], self.key_scales[layer_idx], self.key_zeros[layer_idx], self.n_kv_bits[layer_idx])
                self.value_cache[layer_idx], self.value_scales[layer_idx], self.value_zeros[layer_idx] = self.bit_width_shrinking(self.value_cache[layer_idx], self.value_scales[layer_idx], self.value_zeros[layer_idx], self.n_kv_bits[layer_idx])
                self.n_kv_bits[layer_idx] //= 2

    def dequantize(self, layer_idx, device):
        if self.n_sink_token_in_memory[layer_idx] > 0:
            dequantized_sink_token_key_cache = self.quantizers[self.n_sink_token_bits].dequant(self.sink_token_key_cache[layer_idx], self.sink_token_key_scales[layer_idx], self.sink_token_key_zeros[layer_idx])
            dequantized_sink_token_value_cache = self.quantizers[self.n_sink_token_bits].dequant(self.sink_token_value_cache[layer_idx], self.sink_token_value_scales[layer_idx], self.sink_token_value_zeros[layer_idx])
        else:
            dequantized_sink_token_key_cache = torch.zeros(0, dtype=torch.bfloat16, device=device)
            dequantized_sink_token_value_cache = torch.zeros(0, dtype=torch.bfloat16, device=device)
        if self.n_window_token_in_memory[layer_idx] > 0:
            dequantized_window_token_key_cache = self.quantizers[self.n_window_token_bits].dequant(self.window_token_key_cache[layer_idx], self.window_token_key_scales[layer_idx], self.window_token_key_zeros[layer_idx])
            dequantized_window_token_value_cache = self.quantizers[self.n_window_token_bits].dequant(self.window_token_value_cache[layer_idx], self.window_token_value_scales[layer_idx], self.window_token_value_zeros[layer_idx])
        else:
            dequantized_window_token_key_cache = torch.zeros(0, dtype=torch.bfloat16, device=device)
            dequantized_window_token_value_cache = torch.zeros(0, dtype=torch.bfloat16, device=device)
        if self.key_cache[layer_idx].shape[-2] > 0:
            dequantized_key_cache = self.quantizers[self.n_kv_bits[layer_idx]].dequant(self.key_cache[layer_idx], self.key_scales[layer_idx], self.key_zeros[layer_idx])
            dequantized_value_cache = self.quantizers[self.n_kv_bits[layer_idx]].dequant(self.value_cache[layer_idx], self.value_scales[layer_idx], self.value_zeros[layer_idx])
        else:
            dequantized_key_cache = torch.zeros(0, dtype=torch.bfloat16, device=device)
            dequantized_value_cache = torch.zeros(0, dtype=torch.bfloat16, device=device)

        keys_to_return = torch.cat([dequantized_sink_token_key_cache, dequantized_key_cache, dequantized_window_token_key_cache], dim=-2)
        values_to_return = torch.cat([dequantized_sink_token_value_cache, dequantized_value_cache, dequantized_window_token_value_cache], dim=-2)

        return keys_to_return, values_to_return

    @staticmethod
    def bit_width_shrinking(tensor, scales, zeros, n_original_bits):
        tensor = unpack_funcs[n_original_bits](tensor)
        if n_original_bits == 16:
            tensor = tensor.to(torch.int32)
        tensor = tensor & (2**n_original_bits - 1)
        n_current_bits = n_original_bits // 2
        tensor = ((((tensor << (2 * n_current_bits)) - (tensor << (n_current_bits)) + tensor) + (2 ** (3 * n_current_bits - 1) - 2 ** (2 * n_current_bits - 1) + 2 ** (n_current_bits - 1))) >> (3 * n_current_bits)) & (2**n_current_bits - 1)
        scales *= 2**n_current_bits + 1
        if n_original_bits == 16:
            tensor = tensor.to(torch.int16)
        tensor = pack_funcs[n_current_bits](tensor)
        return tensor, scales, zeros

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        n_sink_token_in_memory = 0 if layer_idx == len(self.n_sink_token_in_memory) else self.n_sink_token_in_memory[layer_idx]
        n_window_token_in_memory = 0 if layer_idx == len(self.n_window_token_in_memory) else self.n_window_token_in_memory[layer_idx]
        n_key_values = 0 if layer_idx == len(self.key_cache) else self.key_cache[layer_idx].shape[-2]
        return n_sink_token_in_memory + n_key_values + n_window_token_in_memory

    def get_max_cache_shape(self) -> Optional[int]:
        return None
