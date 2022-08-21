import logging
import os

logging.basicConfig()

MILVUS_HOST = (os.environ.get("MILVUS_HOST", "localhost"),)
MILVUS_PORT = (os.environ.get("MILVUS_PORT", "19530"),)
