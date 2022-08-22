import logging
from typing import Optional

from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

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
