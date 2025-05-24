from datasets import load_dataset, Dataset


def get_calib_redpajama(path, n_samples, seq_len, tokenizer, seed=42):
    dataset = load_dataset(path, split="train").select(range(1510)).remove_columns("meta").shuffle(seed=seed)
    all_tokens = []
    for example in dataset:
        text = example["text"]
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(tokens)
        if len(all_tokens) > n_samples * seq_len:
            break

    total_samples = len(all_tokens) // seq_len
    samples = [{"input_ids": all_tokens[i * seq_len : (i + 1) * seq_len]} for i in range(total_samples)]

    if n_samples is not None:
        samples = samples[:n_samples]

    return Dataset.from_list(samples)
