"""
Утилита для загрузки CSV (train.csv соревнования TETA) в Kafka.

Пример (без заголовка, под формат CSV):
python3 clickhouse/load_to_kafka.py \
  --csv-path ~/Downloads/train.csv \
  --bootstrap kafka:9092 \
  --topic teta_transactions \
  --batch-size 5000
"""

from __future__ import annotations

import argparse
import csv
import io
import time
from pathlib import Path
from typing import Iterable

from kafka import KafkaProducer


def send_batch(producer: KafkaProducer, topic: str, rows: Iterable[str]) -> None:
    for line in rows:
        producer.send(topic, value=line)
    producer.flush()


def stream_csv_to_kafka(
    csv_path: Path,
    bootstrap: str,
    topic: str,
    batch_size: int = 1000,
    sleep_seconds: float = 0.0,
    send_header: bool = False,
    max_rows: int | None = None,
) -> None:
    producer = KafkaProducer(
        bootstrap_servers=[bootstrap],
        acks="all",
        linger_ms=50,
        value_serializer=lambda v: v.encode("utf-8"),
    )

    sent = 0
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        if send_header:
            buf = io.StringIO()
            csv.writer(buf).writerow(header)
            send_batch(producer, topic, [buf.getvalue().strip("\r\n")])

        buffer: list[str] = []
        for row in reader:
            if max_rows is not None and sent + len(buffer) >= max_rows:
                break
            buf = io.StringIO()
            csv.writer(buf).writerow(row)
            buffer.append(buf.getvalue().strip("\r\n"))
            if len(buffer) >= batch_size:
                send_batch(producer, topic, buffer)
                sent += len(buffer)
                buffer.clear()
                if sleep_seconds:
                    time.sleep(sleep_seconds)

        if buffer:
            if max_rows is not None:
                buffer = buffer[: max_rows - sent]
            send_batch(producer, topic, buffer)
            sent += len(buffer)

    producer.close()
    print(f"Send completed: {sent} rows -> topic {topic}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream CSV file into Kafka topic.")
    parser.add_argument("--csv-path", required=True, help="Путь до train.csv или другого CSV.")
    parser.add_argument("--bootstrap", default="localhost:9092", help="Адрес Kafka bootstrap сервера.")
    parser.add_argument("--topic", default="teta_transactions", help="Имя Kafka топика.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Размер пакета отправки.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Пауза между пакетами.")
    parser.add_argument("--send-header", action="store_true", help="Отправить строку с именами колонок первой.")
    parser.add_argument("--max-rows", type=int, help="Отправить не более N строк (для быстрой проверки).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stream_csv_to_kafka(
        csv_path=Path(args.csv_path),
        bootstrap=args.bootstrap,
        topic=args.topic,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        send_header=args.send_header,
        max_rows=args.max_rows,
    )
