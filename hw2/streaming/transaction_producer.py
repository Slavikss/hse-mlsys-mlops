import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable
from uuid import uuid4

import pandas as pd

from .config import AppConfig
from .kafka_utils import create_producer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("transaction-producer")


def load_transactions(csv_path: Path) -> Iterable[Dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Transactions file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    LOGGER.info("Loaded %s transactions from %s", len(records), csv_path)
    return records


def ensure_transaction_id(record: Dict) -> Dict:
    for key in ("transaction_id", "transactionId", "index", "id", "row_id"):
        if key in record and record[key] not in (None, ""):
            return record
    record = dict(record)
    record["transaction_id"] = str(uuid4())
    return record


def main() -> None:
    cfg = AppConfig()
    csv_path = Path(os.getenv("TRANSACTIONS_FILE", "input/test.csv"))
    sleep_seconds = float(os.getenv("PRODUCER_SLEEP_SECONDS", "1"))
    repeat = os.getenv("PRODUCER_REPEAT", "true").lower() in {"1", "true", "yes", "y"}

    producer = create_producer(cfg.kafka)
    LOGGER.info(
        "Starting producer for topic '%s' with file '%s' (repeat=%s)",
        cfg.kafka.input_topic,
        csv_path,
        repeat,
    )

    while True:
        for record in load_transactions(csv_path):
            enriched = ensure_transaction_id(record)
            producer.send(cfg.kafka.input_topic, value=enriched)
            LOGGER.info(
                "Sent transaction_id=%s to topic %s",
                enriched.get("transaction_id"),
                cfg.kafka.input_topic,
            )
            time.sleep(sleep_seconds)

        producer.flush()

        if not repeat:
            LOGGER.info("Finished sending transactions once; exiting.")
            break

        LOGGER.info("Sleeping 5 seconds before next iteration")
        time.sleep(5)


if __name__ == "__main__":
    main()

