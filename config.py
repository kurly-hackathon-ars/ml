import logging
import os

import mysql.connector
import pymilvus

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("parso").setLevel(logging.INFO)

MILVUS_HOST = os.environ.get("MILVUS_HOST", "localhost")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_COLLECTION_ITEMS = "items"
MILVUS_DEFAULT_ALIAS = "default"

milvus_conn = pymilvus.connections.connect(
    MILVUS_DEFAULT_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT
)


LAZY_LOAD_EXTRACTOR_PIPELINE = (
    os.environ.get("LAZY_LOAD_EXTRACTOR_PIPELINE", "true").lower() == "true"
)

MYSQL_HOST = os.environ.get(
    "MYSQL_HOST", "ars.comk6y2dtzgi.us-west-2.rds.amazonaws.com"
)
MYSQL_DATABASE = "kurly"
MYSQL_USERNAME = "root"
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "ars!!kurly#")

mysql_connection = mysql.connector.connect(
    user=MYSQL_USERNAME,
    host=MYSQL_HOST,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE,
)
