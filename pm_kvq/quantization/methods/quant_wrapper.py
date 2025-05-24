def quantize_model(model, method, method_kwargs):
    if method == "original":
        return

    elif method == "kivi":
        from pm_kvq.quantization.methods.kivi.apply_kivi import apply_kivi

        method_kwargs["k_config"] = {"n_bits": method_kwargs.pop("k_bits"), "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False}
        method_kwargs["v_config"] = {"n_bits": method_kwargs.pop("v_bits"), "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False}
        apply_kivi(model, **method_kwargs)

    elif method == "mikv":
        from pm_kvq.quantization.methods.mikv.apply_mikv import apply_mikv

        method_kwargs["k_config"] = {"n_bits": method_kwargs.pop("k_bits"), "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False}
        method_kwargs["v_config"] = {"n_bits": method_kwargs.pop("v_bits"), "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False}
        apply_mikv(model, **method_kwargs)

    elif method == "rotatekv":
        from pm_kvq.quantization.methods.rotatekv.apply_rotatekv import apply_rotatekv

        method_kwargs["k_config"] = {"n_bits": method_kwargs.pop("k_bits"), "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False}
        method_kwargs["v_config"] = {"n_bits": method_kwargs.pop("v_bits"), "granularity": "per_group", "group_size": 128, "symmetric": False, "round_zeros": False}
        apply_rotatekv(model, **method_kwargs)

    elif method == "pm-kvq":
        backend = method_kwargs.pop("backend", "fake")
        if backend == "fake":
            from pm_kvq.quantization.methods.pm_kvq.apply_pmkvq import apply_fake_pmkvq

            apply_fake_pmkvq(model, **method_kwargs)
        elif backend == "real":
            from pm_kvq.quantization.methods.pm_kvq.apply_pmkvq import apply_real_pmkvq

            apply_real_pmkvq(model, **method_kwargs)
        else:
            raise ValueError

    else:
        raise NotImplementedError
