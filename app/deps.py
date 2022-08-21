import os
from collections import defaultdict
from functools import wraps
from threading import RLock
from typing import Any, Callable, DefaultDict, Dict, List, Optional, cast

import pandas as pd
from pymilvus import connections
from transformers.pipelines import pipeline

from . import models

_db: DefaultDict[str, Dict[int, Any]] = defaultdict(dict)
_lock = RLock()

_milvus_conn = connections.connect(
    alias="default",
    host=os.environ.get("MILVUS_HOST", "localhost"),
    port=os.environ.get("MILVUS_PORT", "19530"),
)


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
def upsert_item(item_id: int, name: str):
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
def setup_sample_items():
    if "items" in _db:
        _db.pop("items")

    if "activities" in _db:
        _db.pop("activities")

    items = pd.read_csv(os.path.join("./data", "movies.csv"))
    for _, item in items.iterrows():
        upsert_item(item["itemId"], item["title"])  # type: ignore

    ratings = pd.read_csv(os.path.join("./data", "ratings.csv"))
    for _, rating in ratings.iterrows():
        add_activity(
            user_id=rating["userId"],
            item_id=rating["itemId"],
            # activity_type=models.ActivityType.from_rating(rating["rating"]),
            activity_type=rating["rating"],
        )  # type: ignore


def get_vectors(sentences: List[str]) -> List[Any]:
    extractor = pipeline(
        "feature-extraction",
        model="kykim/bert-kor-base",
        tokenizer="kykim/bert-kor-base",
    )

    vectors = extractor(sentences)

    return [vector[0][0] for vector in vectors]  # [CLS] token
