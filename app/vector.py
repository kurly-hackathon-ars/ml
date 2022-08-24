import logging
from typing import List, Optional

import torch
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

logger = logging.getLogger(__name__)


_extractor: Optional[Pipeline] = None

_BASE_MODEL_NAME = "kykim/bert-kor-base"
_TOKENIZER_NAME = "kykim/bert-kor-base"


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
