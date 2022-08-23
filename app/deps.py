import logging
from collections import defaultdict
from functools import wraps
from threading import RLock
from typing import (Any, Callable, DefaultDict, Dict, List, Optional, Union,
                    cast)

import config
import pandas as pd

from app.milvus import MilvusHelper

from . import models, vector

logger = logging.getLogger(__name__)

_GET_VECTORS_BATCH_SIZE = 100

_db: DefaultDict[str, Dict[Union[int, str], Any]] = defaultdict(dict)
_lock = RLock()


def _concurrent_lock(fn: Callable):
    @wraps(fn)
    def _wrapper(*args, **kwargs):
        with _lock:  # TODO: check async
            return fn(*args, **kwargs)

    return _wrapper


def get_items() -> List[models.Item]:
    return [cast(models.Item, item) for _, item in _db["items"].items()]


def get_item_filter_dictionaries() -> List[models.ItemFilterDictionary]:
    return [
        cast(models.ItemFilterDictionary, item)
        for _, item in _db["item_filter_dictionaries"].items()
    ]


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
def upsert_item(item_id: int, name: str, category: str) -> models.Item:
    items = _db["items"]
    items_idx = _db["items_idx"]

    # Update
    if item_id in items:
        item = cast(models.Item, items[item_id])
        item.name = name
        index = item.index
    else:
        index = len(items)
        item = models.Item(index=index, id=item_id, name=name, category=category)

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


@_concurrent_lock
def upsert_item_filter_dictionary(s: str) -> models.ItemFilterDictionary:
    dicts = _db["item_filter_dictionaries"]

    dicts[s] = models.ItemFilterDictionary(keyword=s)
    return dicts[s]


def delete_item(item_id: int):
    try:
        _db["items"].pop(item_id)
    except KeyError:
        ...


@_concurrent_lock
def setup_sample_items_from_csv(
    items_fp: str = "./data/items.csv",
    ratings_fp: str = "./data/ratings.csv",
):
    if "items" in _db:
        _db.pop("items")

    if "activities" in _db:
        _db.pop("activities")

    items = pd.read_csv(items_fp)
    for _, item in items.iterrows():
        upsert_item(item["itemId"], item["title"], item["category"])

    ratings = pd.read_csv(ratings_fp)
    for _, rating in ratings:
        add_activity(rating["userId"], rating["itemId"], activity_type=rating["rating"])


@_concurrent_lock
def setup_sample_items_from_mysql(
    sql_table: str = "kurly_products",  # products(beauty from Olive), kurly_products
    limit: Optional[int] = None,
):
    if "items" in _db:
        _db.pop("items")

    if "activities" in _db:
        _db.pop("activities")

    cursor = config.mysql_connection.cursor()
    sql_query = f"select * from {sql_table}"

    if limit is not None and limit > 0:
        sql_query += f" limit {limit}"

    cursor.execute(sql_query)
    for each in cursor:
        item_id, name, category = each[0], each[2], each[5]
        upsert_item(item_id, name, category)
        logger.debug(
            "Inserted item<id=%d,name=%s,category=%s>", item_id, name, category
        )


def get_vectors(sentences: List[str]) -> List[Any]:
    logger.info("Received %d sentences.", len(sentences))
    n_batch = (len(sentences) // _GET_VECTORS_BATCH_SIZE) + 1

    vectors = []
    for i in range(n_batch):
        logger.debug("[%d/%d] Extracting vector...", i + 1, n_batch)
        vectors.extend(
            vector.get_extractor()(
                sentences[
                    i * _GET_VECTORS_BATCH_SIZE : i * _GET_VECTORS_BATCH_SIZE
                    + _GET_VECTORS_BATCH_SIZE
                ]
            )
        )

    assert len(sentences) == len(vectors), f"{len(sentences)} != {len(vectors)}"
    return [vector[0][0] for vector in vectors]  # [CLS] token


if not config.LAZY_LOAD_EXTRACTOR_PIPELINE:
    vector.get_extractor()


def insert_entities(items: List[models.Item]):
    item_ids = [item.id for item in items]
    item_name_vectors = get_vectors([item.name for item in items])
    MilvusHelper.insert(item_ids, item_name_vectors)


def get_training_data():
    cursor = config.mysql_connection.cursor()
    cursor.execute("select * from products")
    for each in cursor:
        print(each)
