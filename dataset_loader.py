from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
from datasets import load_dataset, DatasetDict

@dataclass
class DataSpec:
    text_key: Optional[str] = None
    # For binary/regression tasks: a single score or label key
    label_key: Optional[str] = None
    label_from_score: bool = False
    threshold: float = 0.5
    # For multilabel tasks: map label name -> column key (e.g., {"toxicity": "toxicity", "insult": "insult"})
    multilabel_map: Optional[Dict[str, str]] = None
    # Other hints
    join_messages: bool = False

# Known, concrete schemas (confirmed via HF viewer/cards)
REGISTRY: Dict[str, DataSpec] = {
    # AllenAI RealToxicityPrompts: `text` + scores like `toxicity`, `severe_toxicity`, `insult`, ...
    "allenai/real-toxicity-prompts": DataSpec(text_key="text", label_key="toxicity", label_from_score=True, threshold=0.5),
    # Some Innodata RT variants expose free-text under `text` or chat `messages`. Keep flexible.
    "innodatalabs/rt-realtoxicity-translation": DataSpec(text_key=None, label_key=None),
    "innodatalabs/rt-inod-bias": DataSpec(text_key=None, label_key=None),
}


_TEXT_CANDIDATES = ["text", "prompt", "content", "question", "sentence", "response"]
_LABEL_CANDIDATES = ["label", "labels", "target", "y", "class"]


def _detect_text_and_label(example: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    keys = list(example.keys())
    # Prefer obvious string fields
    for k in _TEXT_CANDIDATES:
        if k in example and isinstance(example[k], str):
            return k, None, {}
    # Chat-style messages
    if "messages" in example and isinstance(example["messages"], list):
        return "messages", None, {"join_messages": True}
    # Fallback: first str field
    for k in keys:
        if isinstance(example[k], str):
            return k, None, {}
    return None, None, {}

def _join_messages(msgs: List[Any]) -> str:
    parts = []
    for m in msgs:
        if isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content")
            if content:
                parts.append(f"{role}: {content}")
        elif isinstance(m, str):
            parts.append(m)
    return "\n".join(parts)


def _extract_text(example: Dict[str, Any], spec: DataSpec) -> str:
    if spec.text_key and spec.text_key in example:
        v = example[spec.text_key]
    else:
        tk, _, extra = _detect_text_and_label(example)
        spec.text_key = tk
        spec.join_messages = extra.get("join_messages", False)
        v = example.get(tk)

    if spec.join_messages and isinstance(v, list):
        return _join_messages(v)
    return v if isinstance(v, str) else str(v)


def _extract_label(example: Dict[str, Any], spec: DataSpec, task: str, score_key: Optional[str], thr: float):
    # Multilabel: collect listed columns
    if task == "multilabel" and spec.multilabel_map:
        return {name: int(float(example[col]) >= thr) for name, col in spec.multilabel_map.items() if col in example}


    # Explicit label column
    if spec.label_key and spec.label_key in example:
        val = example[spec.label_key]
        if task == "binary" and (spec.label_from_score or isinstance(val, (float, int))):
            return int(float(val) >= thr)
        if task == "regression" and isinstance(val, (float, int)):
            return float(val)
        # Already categorical/int label
        return int(val) if isinstance(val, (int, float)) else val


    # Task-directed with provided score_key
    if score_key and score_key in example:
        val = example[score_key]
        return int(float(val) >= thr) if task == "binary" else float(val)


    # Common label names
    for k in _LABEL_CANDIDATES:
        if k in example:
            v = example[k]
            return int(v) if isinstance(v, (int, float)) else v
    return None

def _standardize(dsdict: DatasetDict, spec: DataSpec, name: str, task: str, score_key: Optional[str], thr: float) -> DatasetDict:
    def map_fn(ex):
        text = _extract_text(ex, spec)
        label = _extract_label(ex, spec, task, score_key, thr)
        out = {"text": text}
        if label is not None:
            out["label"] = label
        return out
    # Remove all but text/label to keep the footprint small
    keep = {"text", "label"}
    remove = [c for c in dsdict[list(dsdict.keys())[0]].column_names if c not in keep]


    return dsdict.map(map_fn, remove_columns=remove, desc=f"Standardizing {name}")



def _ensure_splits(dsdict: DatasetDict, seed: int = 42) -> DatasetDict:
    if "train" not in dsdict:
        k0 = list(dsdict.keys())[0]
        dsdict = DatasetDict({"train": dsdict[k0]})
    if "validation" not in dsdict or "test" not in dsdict:
        split = dsdict["train"].train_test_split(test_size=0.2, seed=seed)
        dsdict = DatasetDict({"train": split["train"], "test": split["test"]})
        val_split = dsdict["train"].train_test_split(test_size=0.1, seed=seed)
        dsdict["train"], dsdict["validation"] = val_split["train"], val_split["test"]
    return dsdict




def load_and_standardize(
    name: str,
    task: str = "binary", # "binary", "regression", or "multilabel"
    score_key: Optional[str] = None, # used for binary/regression when no explicit label
    threshold: float = 0.5,
    seed: int = 42,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    # Load (will require HF login if dataset is gated)
    ds_any = load_dataset(name)
    dsdict = ds_any if isinstance(ds_any, DatasetDict) else DatasetDict({"train": ds_any})


    # Ensure splits
    dsdict = _ensure_splits(dsdict, seed=seed)


    # Fetch spec or create an empty one
    spec = REGISTRY.get(name, DataSpec())


    # Standardize
    dsdict = _standardize(dsdict, spec, name, task, score_key, threshold)


    info = {
        "source": name,
        "task": task,
        "text_key": spec.text_key,
        "label_key": spec.label_key or score_key,
        "threshold": threshold if task == "binary" else None,
        "join_messages": spec.join_messages,
        "splits": {k: v.num_rows for k, v in dsdict.items()},
    }
    return dsdict, info