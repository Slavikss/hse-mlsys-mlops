import logging
import time
from typing import Dict

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import AppConfig
from .kafka_utils import create_consumer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("score-writer")


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table_name} (
    transaction_id TEXT PRIMARY KEY,
    score DOUBLE PRECISION NOT NULL,
    fraud_flag SMALLINT NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
"""

UPSERT_SQL = """
INSERT INTO {table_name} (transaction_id, score, fraud_flag, processed_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (transaction_id)
DO UPDATE SET score = EXCLUDED.score,
              fraud_flag = EXCLUDED.fraud_flag,
              processed_at = EXCLUDED.processed_at;
"""


def wait_for_db(cfg: AppConfig, retries: int = 10, delay: float = 3.0):
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                host=cfg.database.host,
                port=cfg.database.port,
                dbname=cfg.database.database,
                user=cfg.database.user,
                password=cfg.database.password,
            )
            conn.autocommit = True
            LOGGER.info("Connected to Postgres at %s:%s", cfg.database.host, cfg.database.port)
            return conn
        except psycopg2.OperationalError:
            sleep_for = delay * (attempt + 1)
            LOGGER.warning(
                "Postgres unavailable, retrying in %.1f seconds (attempt %s/%s)",
                sleep_for,
                attempt + 1,
                retries,
            )
            time.sleep(sleep_for)
    raise ConnectionError("Failed to connect to Postgres after multiple attempts")


def ensure_table(conn, table_name: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(CREATE_TABLE_SQL.format(table_name=table_name))
    LOGGER.info("Ensured table '%s' exists", table_name)


def upsert_score(conn, table_name: str, payload: Dict) -> None:
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            UPSERT_SQL.format(table_name=table_name),
            (
                str(payload["transaction_id"]),
                float(payload["score"]),
                int(payload["fraud_flag"]),
            ),
        )
    LOGGER.debug(
        "Upserted transaction_id=%s score=%.4f fraud_flag=%s",
        payload["transaction_id"],
        payload["score"],
        payload["fraud_flag"],
    )


def main() -> None:
    cfg = AppConfig()
    conn = wait_for_db(cfg)
    ensure_table(conn, cfg.database.table_name)

    consumer = create_consumer(
        cfg.kafka,
        topics=[cfg.kafka.output_topic],
        group_id=f"{cfg.kafka.consumer_group}-db-writer",
        enable_auto_commit=False,
    )

    LOGGER.info(
        "Listening for scored transactions on topic '%s' to persist in table '%s'",
        cfg.kafka.output_topic,
        cfg.database.table_name,
    )

    for message in consumer:
        payload = message.value
        try:
            upsert_score(conn, cfg.database.table_name, payload)
            consumer.commit()
            LOGGER.info(
                "Stored score for transaction_id=%s flag=%s score=%.4f",
                payload.get("transaction_id"),
                payload.get("fraud_flag"),
                payload.get("score"),
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to persist payload: %s", payload)


if __name__ == "__main__":
    main()

