from pm_kvq.evaluation.eval_aime import eval_aime
from pm_kvq.evaluation.eval_cmimc import eval_cmimc
from pm_kvq.evaluation.eval_livecodebench.eval_livecodebench import eval_livecodebench_code_generation


def evaluate_model(model, tokenizer, dataset, evaluate_kwargs, generate_kwargs):
    if dataset == "aime":
        eval_aime(model, tokenizer, **evaluate_kwargs, **generate_kwargs)
    elif dataset == "cmimc":
        eval_cmimc(model, tokenizer, **evaluate_kwargs, **generate_kwargs)
    elif dataset == "livecodebench":
        eval_livecodebench_code_generation(model, tokenizer, **evaluate_kwargs, **generate_kwargs)
    else:
        raise NotImplementedError
