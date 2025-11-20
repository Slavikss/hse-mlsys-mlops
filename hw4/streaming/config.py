import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@dataclass(frozen=True)
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    input_topic: str = os.getenv("KAFKA_INPUT_TOPIC", "transactions")
    output_topic: str = os.getenv("KAFKA_OUTPUT_TOPIC", "scores")
    consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-scorer")
    auto_offset_reset: str = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
    enable_auto_commit: bool = os.getenv("KAFKA_ENABLE_AUTO_COMMIT", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    client_id: str = os.getenv("KAFKA_CLIENT_ID", "mlops-fraud-service")


@dataclass(frozen=True)
class DatabaseConfig:
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "fraud")
    user: str = os.getenv("POSTGRES_USER", "fraud")
    password: str = os.getenv("POSTGRES_PASSWORD", "fraud")
    table_name: str = os.getenv("POSTGRES_TABLE", "transaction_scores")


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path = Path(os.getenv("MODEL_PATH", "artifacts/catboost_model.cbm"))
    metadata_path: Path = Path(os.getenv("METADATA_PATH", "artifacts/feature_metadata.json"))
    threshold: Optional[float] = (
        float(os.getenv("MODEL_THRESHOLD"))
        if os.getenv("MODEL_THRESHOLD") is not None
        else None
    )

    def load_metadata(self) -> dict:
        return _load_json(self.metadata_path)

    def feature_columns(self) -> List[str]:
        metadata = self.load_metadata()
        columns: List[str] = metadata.get("feature_columns") or metadata.get(
            "numeric_features", []
        )
        cat_columns: List[str] = metadata.get("categorical_features", [])
        if columns and cat_columns:
            return columns
        return cat_columns + metadata.get("numeric_features", [])

    def decision_threshold(self) -> float:
        if self.threshold is not None:
            return self.threshold
        metadata = self.load_metadata()
        return float(metadata.get("decision_threshold", 0.5))


@dataclass(frozen=True)
class AppConfig:
    kafka: KafkaConfig = KafkaConfig()
    database: DatabaseConfig = DatabaseConfig()
    model: ModelConfig = ModelConfig()

