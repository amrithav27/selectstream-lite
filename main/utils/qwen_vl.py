from __future__ import annotations
from typing import Tuple, Any, Dict
from main.constants import ALL_SPECIAL_TOKENS
import torch


def add_tokens(tokenizer):
    special = {"additional_special_tokens": ALL_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special)
    return tokenizer


def init_token_embeddings(model, tokenizer, init_from_token: str | None = None, noise_std: float = 1e-3):
    emb_layer = model.get_input_embeddings()
    if emb_layer is None:
        return
    w_in = emb_layer.weight

    init_id = None
    if init_from_token is not None:
        init_id = tokenizer.convert_tokens_to_ids(init_from_token)
    if init_id is None or init_id == tokenizer.unk_token_id:
        init_id = tokenizer.eos_token_id
    if init_id is None:
        return

    with torch.no_grad():
        ref = w_in[init_id].detach().clone()
        for tok in ALL_SPECIAL_TOKENS:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is None or tid == tokenizer.unk_token_id:
                continue
            w_in[tid].copy_(ref + torch.randn_like(ref) * noise_std)

        out_layer = getattr(model, "get_output_embeddings", lambda: None)()
        if out_layer is not None and out_layer.weight.data_ptr() != w_in.data_ptr():
            w_out = out_layer.weight
            for tok in ALL_SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is None or tid == tokenizer.unk_token_id:
                    continue
                w_out[tid].copy_(w_in[tid])


def _vl_model_class():
    # transformers 5.x removed AutoModelForVision2Seq; AutoModelForCausalLM does not map Qwen-VL.
    import transformers

    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise ImportError("No vision-language auto class found in this transformers version.")


def load_qwen25vl(model_name_or_path: str, torch_dtype=None, device_map="auto", trust_remote_code=True, max_memory=None):
    from transformers import AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    old_vocab = len(tokenizer)
    tokenizer = add_tokens(tokenizer)
    kwargs = {"trust_remote_code": trust_remote_code, "device_map": device_map}
    if max_memory is not None:
        kwargs["max_memory"] = max_memory
    try:
        model = _vl_model_class().from_pretrained(model_name_or_path, dtype=torch_dtype, **kwargs)
    except TypeError:
        model = _vl_model_class().from_pretrained(model_name_or_path, torch_dtype=torch_dtype, **kwargs)
    # resize embeddings after adding tokens
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    if len(tokenizer) > old_vocab:
        init_token_embeddings(model, tokenizer, init_from_token=None, noise_std=1e-3)

    return model, tokenizer, processor
