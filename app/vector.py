import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_metric
from datasets.arrow_dataset import Dataset
from transformers.models.auto.modeling_auto import (
    AutoModel, AutoModelForSequenceClassification)
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


_extractor: Optional[Pipeline] = None

_BASE_MODEL_NAME = "kykim/bert-kor-base"
_TOKENIZER_NAME = "kykim/bert-kor-base"
_TRAIN_OUTPUT_DIR = "./train_output"
_MODEL_FILE_PATH = "./models/trained_model"
# _TOKENIZER_NAME = _MODEL_FILE_PATH


def get_extractor() -> Pipeline:
    global _extractor
    if _extractor is not None:
        return _extractor

    _extractor = pipeline(
        "feature-extraction",
        model=_BASE_MODEL_NAME,
        tokenizer=_TOKENIZER_NAME,
    )

    return _extractor


def get_embedding(sentences: List[str]):
    model = AutoModel.from_pretrained(_BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**encoded_input)

    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    sentence_embeddings = _mean_pooling(output, encoded_input["attention_mask"])
    return sentence_embeddings


def train_model(data: str):
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)

    def _map(d):
        return tokenizer(d["text"], padding="max_length", truncation=True)

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    dataset = Dataset.from_csv(data).train_test_split()  # type: ignore
    dataset = dataset.map(_map, batched=True)

    train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))
    logger.info("Training model with %s...", train_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL_NAME, num_labels=2
    )

    train_args = TrainingArguments(
        output_dir=_TRAIN_OUTPUT_DIR, evaluation_strategy="epoch"
    )
    metric = load_metric("accuracy")

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        compute_metrics=_compute_metrics,  # type: ignore
    )

    trainer.train()
    trainer.save_model(_MODEL_FILE_PATH)

    global _extractor
    _extractor = None
