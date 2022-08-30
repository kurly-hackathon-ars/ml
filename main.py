import csv
import logging
from collections import Counter, defaultdict
from typing import List, Optional

import typer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import config
from app import deps, models, service
from app.milvus import MilvusHelper
from config import *

logger = logging.getLogger(__name__)

app = FastAPI(title="Kurly Festa API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


########################################################################################################################
# Services
########################################################################################################################
@app.get("/health")
def health():
    return {"status": "ok"}


# 사용자의 상품 검색시, 벡터기반의 검색결과를 추가로 노출하여 유사 상품 구매를 제안
@app.post("/recommend_by_keyword", response_model=List[models.RecommendedItem])
def recommend_by_keyword(request: models.RecommendByKeyword):
    return [
        models.RecommendedItem(
            no=each.item.id,
            name=each.item.name,
            category=each.item.category,
            img_url=each.item.img_url,
            origin_price=each.item.origin_price,
            sale_price=each.item.sale_price,
        )
        for each in service.recommend_by_vector(request.keyword)[:20]
    ]


_TYPE_TO_RET_MAP = {
    "PURCHASE": "buy",
    "CART": "cart",
    "FAVORITE": "like",
    "CLICK": "view",
    "SEARCH": "search",
}

# 사용자의 행동 데이터 (좋아요, 장바구니 담기, 최근 상품 등)들을 통해 생성, 가공된 상품 데이터를 통해 실시간 추천
@app.get("/recommend_by_activity")
def recommend_by_activity(type: Optional[str] = None):
    """types: buy, cart, like, view, search"""
    # PURCHASE, CART, FAVORITE, CLICK, SEARCH
    type_to_items = defaultdict(list)
    activities = deps.get_activities2()
    logger.info("Found %d activities...", len(activities))
    for row in activities:
        item_id, activity_type = row[1], row[3]
        type_to_items[activity_type].append(item_id)

    type_to_items_ret = defaultdict(list)
    for k, v in type_to_items.items():
        if k not in _TYPE_TO_RET_MAP:
            continue

        for k2, v2 in list(
            sorted(list(Counter(v).items()), key=lambda x: x[1], reverse=True)
        )[:10]:
            type_to_items_ret[_TYPE_TO_RET_MAP[k]].append(deps.get_item_by_id(k2))

    if type:
        return type_to_items_ret[type]

    return type_to_items_ret


@app.get("/items/batch/{item_ids}", tags=["Management"])
def get_items_by_ids(item_ids: str):
    return deps.get_items_by_ids([int(id) for id in item_ids.split(",")])


@app.get("/activities", tags=["Management"])
def get_all_acitivies():
    return deps.get_activities()


@app.put("/items", tags=["Management"])
def put_item(request: models.PutItemRequest):
    service.insert_item(
        request.id,
        request.name,
        request.category,
        request.img_url,
        request.origin_price or 0,
        request.sale_price or 0,
    )
    return Response(status_code=200)


@app.post("/activities", tags=["Management"])
def post_activity(request: models.PostActivityRequest):
    """
    <activity_type 설명>
    1: search
    2: view
    3: like
    4: cart
    5: buy
    """

    deps.add_activity(
        user_id=request.user_id,
        item_id=request.item_id,
        activity_type=request.activity_type.value,
    )
    # service._train_model()
    return Response(status_code=200)


@app.delete("/items/{item_id}", tags=["Management"])
def delete_item(item_id: int):
    deps.delete_item(item_id)
    return Response(status_code=204)


@app.post("/items/setup_samples", tags=["Management"])
def setup_sample_items_for_testing():
    deps.setup_sample_items_from_csv()
    return Response(status_code=200)


########################################################################################################################
# CLI
########################################################################################################################
typer_app = typer.Typer()


@typer_app.command()
def setup_sample_items_from_mysql(limit: int = 200):
    deps.setup_sample_items_from_mysql("kurly_products", limit)


@typer_app.command()
def insert_milvus_entities(fp: str = "./data/items.csv"):
    with open(fp, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        item_ids = []
        titles = []
        for each in reader:
            item_ids.append(int(each["itemId"]))
            titles.append(each["title"])

        vectors = deps.get_vectors(titles)

        MilvusHelper.get_collection().insert([item_ids, vectors])

    collection = MilvusHelper.get_collection()
    collection.load()
    logger.info(
        collection.search(
            deps.get_vectors(["식물"]),
            "item_name",
            {"metric_type": "L2", "params": {"nprobe": 10}},
            limit=10,
            output_fields=["item_id"],
        )
    )


@typer_app.command()
def create_ml_mysql_tables():
    conn = config.get_ml_connection()
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS items")
    cursor.execute(
        "CREATE TABLE items (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), item_id INT UNIQUE, category VARCHAR(255), img_url VARCHAR(2083), origin_price INT, sale_price INT )"
    )
    cursor.execute("DROP TABLE IF EXISTS activities")
    cursor.execute(
        "CREATE TABLE activities (id INT AUTO_INCREMENT PRIMARY KEY, item_id INT, offset INT UNIQUE, activity_type VARCHAR(255))"
    )
    cursor.close()
    conn.close()


@typer_app.command()
def drop_milvus_collection():
    MilvusHelper.get_collection().drop()


if __name__ == "__main__":
    typer_app()
