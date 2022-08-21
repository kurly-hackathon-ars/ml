from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app import deps, models, service
from config import *

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
@app.get("/recommend_by_keyword/{keyword}")
def recommend_by_keyword(keyword: str):
    return [item for item in deps.get_items()[:5]]


# 사용자의 행동 데이터 (좋아요, 장바구니 담기, 최근 상품 등)들을 통해 생성, 가공된 상품 데이터를 통해 실시간 추천
@app.get("/recommend_by_activity/{item_id}", response_model=models.RecommendResponse)
def recommend_by_activity(item_id: int):
    return models.RecommendResponse(
        items=[
            models.RecommendedItem(
                id=item[0],
                score=item[1],
                name=item[2],
            )
            for item in service.recommend_by_activity(item_id)
        ]
    )


########################################################################################################################
# Managements
########################################################################################################################
@app.get("/items", tags=["Management"])
def get_all_items():
    return deps.get_items()


@app.get("/items/{item_id}", tags=["Management"])
def get_item_by_id(item_id: int):
    return deps.get_item_by_id(item_id)


@app.get("/activities", tags=["Management"])
def get_all_acitivies():
    return deps.get_activities()


@app.put("/items", tags=["Management"])
def put_item(request: models.PutItemRequest):
    deps.upsert_item(item_id=request.id, name=request.name)
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
    service._train_model()
    return Response(status_code=200)


@app.delete("/items/{item_id}", tags=["Management"])
def delete_item(item_id: int):
    deps.delete_item(item_id)
    service._train_model()
    return Response(status_code=204)


@app.post("/items/setup_samples", tags=["Management"])
def setup_sample_items_for_testing():
    deps.setup_sample_items()
    service._train_model()
    return Response(status_code=200)
