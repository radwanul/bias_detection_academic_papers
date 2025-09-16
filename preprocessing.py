from typing import Optional
import re
from datasets import DatasetDict
from transformers import AutoTokenizer


_URL_RE = re.compile(r"http\S+|www\.\S+", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")




def basic_clean(text: str) -> str:
    """Conservative cleanup for academic text."""
    if not isinstance(text, str):
        text = str(text)
    text = _BR_RE.sub(" ", text)
    text = _URL_RE.sub(" URL ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text




def get_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)




def tokenize_dataset(
    dsdict: DatasetDict,
    tokenizer=None,
    text_col: str = "text",
    max_length: int = 256,
    pad_to_max: bool = False,
):
    if tokenizer is None:
        tokenizer = get_tokenizer()


    def _batch(examples):
        texts = [basic_clean(t) for t in examples[text_col]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length" if pad_to_max else "longest",
            max_length=max_length,
        )


    tokenized = dsdict.map(_batch, batched=True, desc="Tokenizing")
    cols = ["input_ids", "attention_mask"]
    if "label" in tokenized["train"].column_names:
        cols.append("label")
    tokenized.set_format(type="torch", columns=cols)
    return tokenized




def save_processed(tokenized: DatasetDict, out_dir: str = "data/processed") -> str:
    tokenized.save_to_disk(out_dir)
    return out_dir