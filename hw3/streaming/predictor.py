import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from catboost import CatBoostClassifier

from source.train.preprocess_data import build_features


@dataclass
class FraudPredictor:
    model_path: Path
    feature_columns: List[str]
    threshold: float

    def __post_init__(self) -> None:
        self._model = CatBoostClassifier()
        self._model.load_model(str(self.model_path))

    def predict_row(self, payload: Dict) -> Dict[str, float]:
        raw_df = pd.DataFrame([payload])
        features = build_features(raw_df)

        missing = set(self.feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features after preprocessing: {sorted(missing)}")

        features = features[self.feature_columns]
        probabilities = self._model.predict_proba(features)[:, 1]
        score = float(probabilities[0])
        fraud_flag = int(score >= self.threshold)
        return {"score": score, "fraud_flag": fraud_flag}

    def metadata_json(self) -> str:
        data = {
            "model_path": str(self.model_path),
            "threshold": self.threshold,
            "feature_columns": self.feature_columns,
        }
        return json.dumps(data, ensure_ascii=False)

