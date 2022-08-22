import logging
import os
from collections import defaultdict
from functools import wraps
from threading import RLock
from typing import Any, Callable, DefaultDict, Dict, List, Optional, cast

import config
import pandas as pd
from transformers.pipelines import pipeline
from transformers.pipelines.base import Pipeline

from app.milvus import MilvusHelper

from . import models

logger = logging.getLogger(__name__)

_GET_VECTORS_BATCH_SIZE = 100

_db: DefaultDict[str, Dict[int, Any]] = defaultdict(dict)
_lock = RLock()

_extractor: Optional[Pipeline] = None


def _concurrent_lock(fn: Callable):
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        with _lock:  # TODO: check async
            return fn(*args, **kwargs)

    return _wrapper


def get_items() -> List[models.Item]:
    return [cast(models.Item, item) for _, item in _db["items"].items()]


def get_item_by_id(item_id: int) -> Optional[models.Item]:
    if item_id not in _db["items"]:
        return None

    return _db["items"][item_id]


def get_item_by_index(idx: int) -> Optional[models.Item]:
    if idx not in _db["items_idx"]:
        return None

    return _db["items_idx"][idx]


def get_activities() -> List[models.Activity]:
    return [cast(models.Activity, item) for _, item in _db["activities"].items()]


@_concurrent_lock
def upsert_item(item_id: int, name: str) -> models.Item:
    items = _db["items"]
    items_idx = _db["items_idx"]

    # Update
    if item_id in items:
        item = cast(models.Item, items[item_id])
        item.name = name
        index = item.index
    else:
        index = len(items)
        item = models.Item(index=index, id=item_id, name=name)

    items[item_id] = item
    items_idx[index] = item
    return item


@_concurrent_lock
def add_activity(
    user_id: int,
    item_id: int,
    # activity_type: models.ActivityType,
    activity_type: float,
):
    activities = _db["activities"]
    activity_id = len(activities)
    activity = models.Activity(
        id=activity_id,
        user_id=user_id,
        item_id=item_id,
        activity_type=activity_type,
    )
    activities[activity_id] = activity


def delete_item(item_id: int):
    try:
        _db["items"].pop(item_id)
    except KeyError:
        ...


@_concurrent_lock
def setup_sample_items(
    items_fp: str = "./data/items.csv", ratings_fp: str = "./data/ratings.csv"
):
    if "items" in _db:
        _db.pop("items")

    if "activities" in _db:
        _db.pop("activities")

    items = pd.read_csv(items_fp)
    for _, item in items.iterrows():
        upsert_item(item["itemId"], item["title"])  # type: ignore

    ratings = pd.read_csv(ratings_fp)
    for _, rating in ratings.iterrows():
        add_activity(
            user_id=rating["userId"],
            item_id=rating["itemId"],
            # activity_type=models.ActivityType.from_rating(rating["rating"]),
            activity_type=rating["rating"],
        )  # type: ignore


def get_vectors(sentences: List[str]) -> List[Any]:
    logger.info("Received %d sentences.", len(sentences))
    n_batch = (len(sentences) // _GET_VECTORS_BATCH_SIZE) + 1

    vectors = []
    for i in range(n_batch):
        logger.debug("[%d/%d] Extracting vector...", i + 1, n_batch)
        vectors.extend(
            _get_extractor()(
                sentences[
                    i * _GET_VECTORS_BATCH_SIZE : i * _GET_VECTORS_BATCH_SIZE
                    + _GET_VECTORS_BATCH_SIZE
                ]
            )
        )

    assert len(sentences) == len(vectors), f"{len(sentences)} != {len(vectors)}"
    return [vector[0][0] for vector in vectors]  # [CLS] token


def _get_extractor() -> Pipeline:
    global _extractor
    if _extractor is not None:
        return _extractor

    _extractor = pipeline(
        "feature-extraction",
        model="kykim/bert-kor-base",
        tokenizer="kykim/bert-kor-base",
    )
    return _extractor


if not config.LAZY_LOAD_EXTRACTOR_PIPELINE:
    _get_extractor()


def insert_entities(items: List[models.Item]):
    item_ids = [item.id for item in items]
    item_name_vectors = get_vectors([item.name for item in items])
    MilvusHelper.insert(item_ids, item_name_vectors)


def get_training_data():
    cursor = config.mysql_connection.cursor()
    cursor.execute("select * from products")
    for each in cursor:
        print(each)
