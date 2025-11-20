import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Dict
from uuid import uuid4

from kafka import KafkaConsumer, KafkaProducer

from .config import AppConfig
from .kafka_utils import create_consumer, create_producer
from .predictor import FraudPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("fraud-inference-service")


def resolve_transaction_id(record: Dict) -> str:
    for key in ("transaction_id", "transactionId", "index", "id", "row_id"):
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return str(uuid4())


def _attach_signal_handlers(consumer: KafkaConsumer, producer: KafkaProducer) -> None:
    def _shutdown_handler(signum, frame):  # type: ignore[override]
        LOGGER.info("Received signal %s, shutting down gracefully", signum)
        try:
            consumer.close()
        finally:
            producer.flush()
            producer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)


def main() -> None:
    cfg = AppConfig()
    LOGGER.info("Booting inference service with config: %s", cfg)
    predictor = FraudPredictor(
        model_path=cfg.model.model_path,
        feature_columns=cfg.model.feature_columns(),
        threshold=cfg.model.decision_threshold(),
    )
    LOGGER.info("Loaded model metadata: %s", predictor.metadata_json())

    producer = create_producer(cfg.kafka)
    consumer = create_consumer(
        cfg.kafka,
        topics=[cfg.kafka.input_topic],
        group_id=cfg.kafka.consumer_group,
        enable_auto_commit=False,
    )
    _attach_signal_handlers(consumer, producer)

    LOGGER.info(
        "Starting to consume from topic '%s' and produce to '%s'",
        cfg.kafka.input_topic,
        cfg.kafka.output_topic,
    )

    for message in consumer:
        record: Dict = message.value
        transaction_id = resolve_transaction_id(record)
        try:
            prediction = predictor.predict_row(record)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to score transaction_id=%s payload=%s", transaction_id, record)
            consumer.commit()
            continue

        output_payload = {
            "transaction_id": transaction_id,
            "score": prediction["score"],
            "fraud_flag": prediction["fraud_flag"],
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }

        producer.send(
            cfg.kafka.output_topic,
            value=output_payload,
        )
        producer.flush()
        consumer.commit()

        LOGGER.info(
            "Scored transaction_id=%s score=%.4f fraud_flag=%s",
            transaction_id,
            prediction["score"],
            prediction["fraud_flag"],
        )


if __name__ == "__main__":
    main()

