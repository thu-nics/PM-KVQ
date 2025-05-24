from types import MethodType

import torch

from pm_kvq.utils.modeling_utils import get_llm_layers
from pm_kvq.quantization.methods.pm_kvq.smoothattention.apply_smoothattention import apply_smoothattention_rep
from pm_kvq.quantization.methods.pm_kvq.progressive.apply_progressive import apply_progressive
from pm_kvq.quantization.methods.pm_kvq.progressive.progressive_quantized_cache import ProgressiveQuantizedCache, ProgressiveQuantizedCacheConfig
from pm_kvq.utils.chatbot import _sample


def apply_fake_pmkvq(
    model,
    rep_scales,
    kv_budgets,
    n_sink_token,
    n_sink_token_bits,
    n_window_token,
    n_window_token_bits,
    n_init_kv_bits,
):
    apply_smoothattention_rep(model, rep_scales)
    apply_progressive(model, kv_budgets, n_sink_token, n_sink_token_bits, n_window_token, n_window_token_bits, n_init_kv_bits)


def apply_real_pmkvq(
    model,
    rep_scales,
    kv_budgets,
    n_sink_token,
    n_sink_token_bits,
    n_window_token,
    n_window_token_bits,
    n_init_kv_bits,
):
    apply_smoothattention_rep(model, rep_scales)
    if isinstance(kv_budgets, str):
        kv_budgets = torch.load(kv_budgets)
    if isinstance(kv_budgets, float):
        kv_budgets = [kv_budgets] * len(get_llm_layers(model))

    def change_cache_hook(module, args, kwargs):
        if "past_key_values" not in kwargs.keys() or not isinstance(kwargs["past_key_values"], ProgressiveQuantizedCache):
            cache_config = ProgressiveQuantizedCacheConfig(
                kv_budgets=kv_budgets,
                n_sink_token=n_sink_token,
                n_sink_token_bits=n_sink_token_bits,
                n_window_token=n_window_token,
                n_window_token_bits=n_window_token_bits,
                n_init_kv_bits=n_init_kv_bits,
            )
            kwargs["past_key_values"] = ProgressiveQuantizedCache(cache_config)
            return (args, kwargs)

    model.register_forward_pre_hook(change_cache_hook, with_kwargs=True)
    model._sample = MethodType(_sample, model)
