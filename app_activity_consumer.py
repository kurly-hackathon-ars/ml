import logging

from kafka import KafkaConsumer, TopicPartition

import config
from app import deps

logger = logging.getLogger(__name__)

TOPIC_NAME = "ars"


def main():
    logger.info("Starting activity consumer...")
    consumer = KafkaConsumer(bootstrap_servers=["3.37.151.144:9092"])
    tp = TopicPartition(TOPIC_NAME, 0)
    consumer.assign([tp])
    consumer.seek_to_beginning(tp)
    for msg in consumer:
        logger.info("Received <%s> from topic<%s>", msg, TOPIC_NAME)
        data = [
            each.strip().split("=")[-1] for each in msg.value.decode("utf-8").split(",")
        ]
        item_id, action_type = data[2], data[3]
        deps.upsert_activity(int(item_id), action_type, int(msg.offset))

    logger.info("Stopping activity consumer...")


if __name__ == "__main__":
    main()
