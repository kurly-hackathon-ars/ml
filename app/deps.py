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
    cursor = config.ml_mysql_connection.cursor()
    items = []
    cursor.execute("select * from items")
    for each in cursor:
        item = models.Item(id=each[2], index=each[0], name=each[1], category=each[3])
        items.append(item)

    return items


def get_item_filter_dictionaries() -> List[models.ItemFilterDictionary]:
    return [
        cast(models.ItemFilterDictionary, item)
        for _, item in _db["item_filter_dictionaries"].items()
    ]


def get_item_by_id(item_id: int) -> Optional[models.Item]:
    cursor = config.ml_mysql_connection.cursor()
    query = f"""
        SELECT * FROM items WHERE item_id={item_id}
    """
    cursor.execute(query)
    for each in cursor:
        item = models.Item(id=each[2], index=each[0], name=each[1], category=each[3])
        break
    else:
        return None

    return item


def get_item_by_index(idx: int) -> Optional[models.Item]:
    cursor = config.ml_mysql_connection.cursor()
    query = f"""
        SELECT * FROM items WHERE id={idx}
    """
    cursor.execute(query)
    for each in cursor:
        item = models.Item(id=each[2], index=each[0], name=each[1], category=each[3])
        break
    else:
        return None

    return item


def get_activities() -> List[models.Activity]:
    return [cast(models.Activity, item) for _, item in _db["activities"].items()]


@_concurrent_lock
def upsert_item(item_id: int, name: str, category: str) -> models.Item:
    cursor = config.ml_mysql_connection.cursor()
    name = name.replace("'", "")
    category = category.replace("'", "")
    query = f"""
        REPLACE INTO items (item_id, name, category)
        VALUES ('{item_id}', '{name}', '{category}')
    """
    cursor.execute(query)
    cursor.fetchall()
    config.ml_mysql_connection.commit()
    return get_item_by_id(item_id)  # type: ignore


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
