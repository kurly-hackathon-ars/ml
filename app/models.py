from enum import Enum
from typing import List

import pydantic


class ActivityType(Enum):
    NO = 0
    SEARCH = 1
    VIEW = 2
    LIKE = 3
    CART = 4
    BUY = 5

    @classmethod
    def from_rating(cls, rating: int):
        for each in cls:
            if each.value == rating:
                return each

        raise ValueError(f"Invalid Rating: {rating}")


class Item(pydantic.BaseModel):
    id: int
    index: int
    name: str
    category: str


class Activity(pydantic.BaseModel):
    id: int
    user_id: int
    item_id: int
    # activity_type: ActivityType
    activity_type: float


class ItemFilterDictionary(pydantic.BaseModel):
    """For Vector search, exclude some keywords like brands, common things in items, ..."""

    keyword: str


class PutItemRequest(Item, pydantic.BaseModel):
    ...


class PostActivityRequest(pydantic.BaseModel):
    user_id: int
    item_id: int
    activity_type: ActivityType


class RecommendByActivityRequest(pydantic.BaseModel):
    item_id: int


class RecommendedItem(pydantic.BaseModel):
    id: int
    name: str
    score: float


class RecommendResponse(pydantic.BaseModel):
    items: List[RecommendedItem]


class MilvusSearchResult(pydantic.BaseModel):
    item_id: int
    item_name: str
    distance: float
