import logging

from kafka import KafkaConsumer

import config

logger = logging.getLogger(__name__)

TOPIC_NAME = "ars"


def main():
    consumer = KafkaConsumer(
        TOPIC_NAME, bootstrap_servers=f"{config.KAFKA_HOST}:{config.KAFKA_PORT}"
    )
    for msg in consumer:
        logger.info("Received <%s> from topic<%s>", msg, TOPIC_NAME)


if __name__ == "__main__":
    main()
