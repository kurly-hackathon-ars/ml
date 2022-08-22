from threading import Lock
from typing import List, Optional

import config
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      SearchResult)


class MilvusHelper(object):
    VECTOR_FIELD_NAME = "item_name"
    SEARCH_LIMIT = 20

    _loaded_collection: Optional[Collection] = None
    _load_lock: Lock = Lock()

    @classmethod
    def get_collection(cls) -> Collection:
        if cls._loaded_collection is None:
            with cls._load_lock:
                cls._loaded_collection = cls._get_collection()
                cls._loaded_collection.load()

        return cls._loaded_collection

    @classmethod
    def _get_collection(cls):
        item_id = FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True)
        item_name = FieldSchema(name="item_name", dtype=DataType.FLOAT_VECTOR, dim=768)
        collection_scheme = CollectionSchema(
            fields=[item_id, item_name], description="items"
        )
        return Collection(
            name=config.MILVUS_COLLECTION_ITEMS,
            schema=collection_scheme,
            using=config.MILVUS_DEFAULT_ALIAS,
            consistency_level="Strong",
        )

    @classmethod
    def search(cls, query_vectors) -> SearchResult:
        collection = cls.get_collection()
        return collection.search(
            query_vectors,
            cls.VECTOR_FIELD_NAME,
            {"metric_type": "L2"},
            limit=cls.SEARCH_LIMIT,
            output_fields=["item_id"],
        )  # type: ignore

    @classmethod
    def insert(cls, item_ids: List[int], item_names):
        cls.get_collection().insert([item_ids, item_names])
        cls._loaded_collection = None

    @classmethod
    def drop(cls):
        cls.get_collection().drop()
        cls._loaded_collection = None
