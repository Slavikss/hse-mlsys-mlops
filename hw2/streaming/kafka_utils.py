import json
import logging
import time
from typing import Iterable, Optional

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from .config import KafkaConfig


def create_producer(
    cfg: KafkaConfig,
    *,
    retries: int = 8,
    base_delay: float = 1.0,
) -> KafkaProducer:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=_split_servers(cfg.bootstrap_servers),
                client_id=cfg.client_id,
                value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                linger_ms=10,
                retries=3,
            )
            logging.info("Connected to Kafka producer at %s", cfg.bootstrap_servers)
            return producer
        except NoBrokersAvailable as exc:
            last_exc = exc
            sleep_for = base_delay * (2**attempt)
            logging.warning(
                "Kafka producer unavailable, retrying in %.1f seconds (attempt %s/%s)",
                sleep_for,
                attempt + 1,
                retries,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def create_consumer(
    cfg: KafkaConfig,
    *,
    topics: Iterable[str],
    group_id: Optional[str] = None,
    enable_auto_commit: Optional[bool] = None,
    retries: int = 8,
    base_delay: float = 1.0,
) -> KafkaConsumer:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=_split_servers(cfg.bootstrap_servers),
                group_id=group_id or cfg.consumer_group,
                auto_offset_reset=cfg.auto_offset_reset,
                enable_auto_commit=(
                    cfg.enable_auto_commit if enable_auto_commit is None else enable_auto_commit
                ),
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            )
            logging.info(
                "Connected to Kafka consumer for topics %s at %s",
                list(topics),
                cfg.bootstrap_servers,
            )
            return consumer
        except NoBrokersAvailable as exc:
            last_exc = exc
            sleep_for = base_delay * (2**attempt)
            logging.warning(
                "Kafka consumer unavailable, retrying in %.1f seconds (attempt %s/%s)",
                sleep_for,
                attempt + 1,
                retries,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def _split_servers(servers: str) -> list[str]:
    return [server.strip() for server in servers.split(",") if server.strip()]

