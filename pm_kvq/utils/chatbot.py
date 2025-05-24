import sys
from copy import deepcopy

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

DEFAULT_GENERATION_KWARGS = {
    "max_new_tokens": 8192,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}

import os
from typing import Optional, Union

from torch import nn
from transformers import Cache, GenerationConfig, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerateNonBeamOutput


class PrintTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.unknown_token = ["�"]
        self.decode_len = 1
        self.max_buffer_len = 5

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0][-self.decode_len :], skip_special_tokens=True)
        if any([token in decoded_text for token in self.unknown_token]):
            self.decode_len += 1
            if self.decode_len > self.max_buffer_len:
                print(decoded_text, end="", flush=True)
                self.decode_len = 1
            return False
        print(decoded_text, end="", flush=True)
        self.decode_len = 1
        return False


def chat(model, tokenizer, text, print_response=False, return_len=False, **kwargs):
    stop_list = StoppingCriteriaList()
    if print_response:
        stop_list.append(PrintTokenStoppingCriteria(tokenizer))
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    inputs_len = inputs["input_ids"].shape[-1]

    generation_kwargs = deepcopy(DEFAULT_GENERATION_KWARGS)
    generation_kwargs.update(kwargs)
    outputs = model.generate(
        inputs["input_ids"],
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stop_list,
        **generation_kwargs,
    )
    decoded_text = tokenizer.decode(outputs[0][inputs_len:], skip_special_tokens=True)
    if return_len:
        outputs_len = len(outputs[0]) - inputs_len
    del inputs, outputs
    torch.cuda.empty_cache()
    if return_len:
        return decoded_text, (inputs_len, outputs_len)
    else:
        return decoded_text


def chatbot(model, tokenizer, **kwargs):
    stop_list = StoppingCriteriaList()
    stop_list.append(PrintTokenStoppingCriteria(tokenizer))
    while True:
        print(">>> ", end="", flush=True)
        prompt = sys.stdin.read()
        if prompt == "over":
            break
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        generation_kwargs = deepcopy(DEFAULT_GENERATION_KWARGS)
        generation_kwargs.update(kwargs)
        outputs = model.generate(
            inputs["input_ids"],
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stop_list,
            **generation_kwargs,
        )
        print()
        del inputs, outputs
        torch.cuda.empty_cache()


def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        is_compileable = is_compileable and not self.generation_config.disable_compile
        if is_compileable and (self.device.type == "cuda" or generation_config.compile_config._compile_all_devices):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    is_prefill = True
    try:
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs
    except MemoryError:
        pass

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
