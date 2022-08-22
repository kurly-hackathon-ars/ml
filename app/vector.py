import logging
from typing import Any, Dict, List, Optional

from datasets.arrow_dataset import Dataset
from transformers.models.auto.modeling_auto import \
    AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline
from transformers.trainer import Trainer

logger = logging.getLogger(__name__)


_extractor: Optional[Pipeline] = None

_MODEL_NAME = "kykim/bert-kor-base"
_TOKENIZER_NAME = "kykim/bert-kor-base"


def get_extractor() -> Pipeline:
    global _extractor
    if _extractor is not None:
        return _extractor

    _extractor = pipeline(
        "feature-extraction",
        model=_MODEL_NAME,
        tokenizer=_TOKENIZER_NAME,
    )
    return _extractor


def train_model(data: str):
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    def _map(d):
        return tokenizer(d["text"], padding="max_length", truncation=True)

    dataset = Dataset.from_csv(data).train_test_split()  # type: ignore
    dataset = dataset.map(_map, batched=True)

    train_dataset = dataset["train"].shuffle(seed=42).select(range(100))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(100))
    logger.info("Training model with %s...", train_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_NAME, num_labels=2
    )
    trainer = Trainer(
        model=model, train_dataset=train_dataset, eval_dataset=eval_dataset  # type: ignore
    )
    trainer.train()
