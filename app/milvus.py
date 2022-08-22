import config
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema


class MilvusHelper(object):
    @classmethod
    def get_collection(cls):
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
