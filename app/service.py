import logging
import re
from typing import List, Set

from app.milvus import MilvusHelper

from . import deps, models

logger = logging.getLogger(__name__)

_COMMON_PUNCTUATIONS_IN_PRODUCT_TITLE = r".,!?\-()[]{}~X "


def recommend_by_vector(query: str) -> List[models.MilvusSearchResult]:
    search_results = MilvusHelper.search(deps.get_vectors([query]))

    results = []
    for hits in search_results:
        for hit in hits:
            item = deps.get_item_by_id(hit.id)
            if not item:
                continue

            results.append(
                models.MilvusSearchResult(
                    distance=hit.distance,
                    item_id=hit.id,
                    item_name=item.name if item else "<unknown>",
                    item=item,
                )
            )
    return results


def insert_item(
    item_id: int,
    item_name: str,
    item_category: str,
    item_img_url: str,
    origin_price: int,
    sale_price: int,
):
    item = deps.upsert_item(
        item_id, item_name, item_category, item_img_url, origin_price, sale_price
    )
    deps.insert_entities([item])


def build_items():
    filter_keywords = set(
        (
            each.keyword.strip()
            for each in deps.get_item_filter_dictionaries()
            if each.keyword.strip()
        )
    )
    items = [item.copy(deep=True) for item in deps.get_items()]
    logger.info("Filter item names by ItemFilterDictionary...")
    for item in items:
        words = re.findall(
            rf"\w+|[{_COMMON_PUNCTUATIONS_IN_PRODUCT_TITLE}\s]", item.name, re.UNICODE
        )

        new_words = []
        for word in words:
            if word.strip() in filter_keywords:
                logger.debug("Drop keyword %s from %s", word, item.name)
                continue

            if word in _COMMON_PUNCTUATIONS_IN_PRODUCT_TITLE:
                logger.debug("Drop punctuation %s from %s", word, item.name)
                continue

            if re.match(r"\d", word):
                logger.debug("Drop number %s from %s", word, item.name)
                continue

            new_words.append(word)

        item.name = " ".join(new_words)
        for each in filter_keywords:
            if each in item.name:
                logger.debug("Drop keyword %s from %s", each, item.name)
                item.name = item.name.replace(each, "")

        logger.info("Processed item name ==> %s", item.name)

    deps.insert_entities(items)


def generate_item_filter_keywords_from_items() -> Set[str]:
    items = deps.get_items()

    keywords = set()
    for item in items:
        for each in re.findall(r"\[\w+\]", item.name, re.UNICODE):
            logger.debug("Found %s from %s", each, item.name)
            keywords.add(each.strip("[").strip("]"))

    return keywords
