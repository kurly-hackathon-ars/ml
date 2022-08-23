from enum import Enum
from typing import List, Optional

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
    img_url: str
    origin_price: Optional[int] = None
    sale_price: Optional[int] = None


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
    no: int
    img_url: str
    name: str
    origin_price: Optional[int] = None
    sale_price: Optional[int] = None
    category: str


class RecommendResponse(pydantic.BaseModel):
    items: List[RecommendedItem]


class MilvusSearchResult(pydantic.BaseModel):
    item_id: int
    item_name: str
    distance: float
    item: Item
