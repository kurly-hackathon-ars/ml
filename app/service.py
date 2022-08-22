import logging
import os
import pickle
from collections import OrderedDict
from typing import Any, List, Optional

import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from app.milvus import MilvusHelper

from . import deps, models

logger = logging.getLogger(__name__)

_MODEL_PATH = "./model2.pkl"
_dataset: Optional[Any] = None
_df: Optional[Any] = None
_knn = None


def recommend_by_vector(query: str) -> List[models.MilvusSearchResult]:
    search_results = MilvusHelper.search(deps.get_vectors([query]))

    results = []
    for hits in search_results:
        for hit in hits:
            item = deps.get_item_by_id(hit.id)
            results.append(
                models.MilvusSearchResult(
                    distance=hit.distance,
                    item_id=hit.id,
                    item_name=item.name if item else "<unknown>",
                )
            )
    return results


def recommend_by_activity(item_id: int):
    distances, indices = _get_recommendation_model().kneighbors(
        _dataset[deps.get_item_by_id(item_id).index], n_neighbors=21
    )
    for index in indices[0]:
        print(deps.get_item_by_index(index))

    rec_movie_indices = sorted(
        list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
        key=lambda x: x[1],
        reverse=True,
    )[:0:-1]

    # Post process
    print(_df)
    df = _df.reset_index()
    item_ids = [df.iloc[each[0]]["item_id"] for each in rec_movie_indices]
    # max_distance = max(each[1] for each in rec_movie_indices)
    # max_score = 1 - (rec_movie_indices[0][1] / max_distance)
    rec_movie_indices = [
        (
            deps.get_item_by_id(int(item_id)).id,
            1.0 - each[1],
            deps.get_item_by_id(int(item_id)).name,
        )
        for each, item_id in zip(rec_movie_indices, item_ids)
        if deps.get_item_by_index(each[0])
    ]
    return rec_movie_indices


def insert_item(item_id: int, item_name: str):
    item = deps.upsert_item(item_id, item_name)
    deps.insert_entities([item])


def _get_recommendation_model():
    global _knn
    if _knn:
        return _knn

    with open(_MODEL_PATH, mode="rb") as f:
        print("Load.........")
        _knn = pickle.load(f)  # TODO: Change Algo
        return _knn


def _get_item(item_id: int):
    global _dataset
    if _dataset is None:
        _set_dataset()

    assert _dataset is not None
    return _dataset[item_id]


def _set_dataset():
    global _dataset
    global _df
    activities = deps.get_activities()

    dataset: List[OrderedDict] = []
    for each in activities:
        data = OrderedDict()
        data["item_id"] = each.item_id
        data["user_id"] = each.user_id
        data["rating"] = each.activity_type
        dataset.append(data)

    if not dataset:
        raise ValueError("No acitivies exist!")

    print(pd.DataFrame(dataset, columns=list(dataset[0].keys())))
    df = pd.DataFrame(dataset, columns=list(dataset[0].keys())).pivot(index="item_id", columns="user_id", values="rating")  # type: ignore
    df.fillna(0, inplace=True)
    print(df)
    _dataset = csr_matrix(df.values)
    _df = df
    print(_dataset)


def _train_model():
    global _knn
    _set_dataset()

    # df = _dataset.reset_index()
    knn = NearestNeighbors(
        metric="cosine", algorithm="brute", n_neighbors=20, n_jobs=-1
    )
    knn.fit(_dataset)
    _knn = knn
    logger.info("Model Trained with %d data.", _dataset.shape[0])
