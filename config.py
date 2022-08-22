import logging
import os

import pymilvus

logging.basicConfig(level=logging.DEBUG)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_COLLECTION_ITEMS = "items"
MILVUS_DEFAULT_ALIAS = "default"

milvus_conn = pymilvus.connections.connect(
    MILVUS_DEFAULT_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT
)
