"""
Data pipeline – tokenisation, block grouping, and DataLoader creation.

Default dataset: EleutherAI/the_pile_deduplicated (streamed from HF Hub).
"""

from typing import Dict, List, Optional, Any, Iterator
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset


def create_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_fn(example: Dict[str, Any], tokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )


def group_texts(
    examples: Dict[str, List],
    block_size: int,
    eos_token_id: int,
) -> Dict[str, List]:
    stream: List[int] = []
    for ids in examples["input_ids"]:
        stream.extend(ids)
        stream.append(eos_token_id)

    total_len = (len(stream) // block_size) * block_size
    if total_len == 0:
        return {"input_ids": [], "labels": []}

    result = {
        "input_ids": [stream[i : i + block_size] for i in range(0, total_len, block_size)]
    }
    result["labels"] = result["input_ids"].copy()
    return result


def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([x["input_ids"] for x in batch], dtype=torch.long),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.long),
    }


def _tokenize_and_group(dataset, tok_fn, grp_fn):
    remove_cols = ["text", "meta"] if "meta" in dataset.column_names else ["text"]
    tokenized = dataset.map(tok_fn, batched=True, remove_columns=remove_cols)
    return tokenized.map(grp_fn, batched=True)


def create_eval_dataloader(
    dataset_name: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    split: str = "validation",
    streaming: bool = True,
    num_workers: int = 0,
    config_name: Optional[str] = None,
):
    from datasets import load_dataset

    tok_fn = partial(tokenize_fn, tokenizer=tokenizer, max_length=block_size)
    grp_fn = partial(group_texts, block_size=block_size, eos_token_id=tokenizer.eos_token_id)

    kw: Dict[str, Any] = {"split": split, "streaming": streaming}
    if config_name:
        kw["name"] = config_name

    ds = load_dataset(dataset_name, **kw)
    lm_ds = _tokenize_and_group(ds, tok_fn, grp_fn)
    return DataLoader(lm_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)


def create_dataloaders(
    dataset_name: str = "EleutherAI/the_pile_deduplicated",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    block_size: int = 2048,
    batch_size: int = 8,
    num_workers: int = 0,
    streaming: bool = True,
    train_split: str = "train",
    eval_split: Optional[str] = None,
    eval_from_train_examples: int = 10000,
) -> tuple:
    from datasets import load_dataset

    tokenizer = create_tokenizer(tokenizer_name)
    eos_token_id = tokenizer.eos_token_id

    train_dataset = load_dataset(dataset_name, split=train_split, streaming=streaming)
    eval_dataset = None

    if eval_split:
        try:
            eval_dataset = load_dataset(dataset_name, split=eval_split, streaming=streaming)
        except ValueError as e:
            if streaming:
                if eval_from_train_examples <= 0:
                    raise ValueError(
                        "eval_from_train_examples must be > 0 when deriving eval from train."
                    ) from e
                eval_dataset = load_dataset(
                    dataset_name, split=train_split, streaming=True
                ).take(eval_from_train_examples)
                train_dataset = train_dataset.skip(eval_from_train_examples)
            else:
                split_ds = train_dataset.train_test_split(test_size=0.01, seed=42, shuffle=True)
                train_dataset = split_ds["train"]
                eval_dataset = split_ds["test"]

    tok_fn = partial(tokenize_fn, tokenizer=tokenizer, max_length=block_size)
    grp_fn = partial(group_texts, block_size=block_size, eos_token_id=eos_token_id)

    lm_train = _tokenize_and_group(train_dataset, tok_fn, grp_fn)
    train_loader = DataLoader(lm_train, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    if eval_dataset is None:
        raise ValueError(
            "Evaluation dataset could not be created. "
            "Set --eval_split or provide a dataset with a valid eval split."
        )

    lm_eval = _tokenize_and_group(eval_dataset, tok_fn, grp_fn)
    eval_loader = DataLoader(lm_eval, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    return train_loader, eval_loader, tokenizer
