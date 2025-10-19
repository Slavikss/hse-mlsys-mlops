import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from source.train.preprocess_data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_features,
)

DEFAULT_PARAM_GRID: List[Dict[str, float]] = [
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
]


def parse_args(cli_args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost model for fraud detection."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("source/data/train.csv"),
        help="Path to raw training CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/catboost_model.cbm"),
        help="Output path for the trained CatBoost model.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Where to store validation metrics (JSON).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("artifacts/feature_metadata.json"),
        help="Where to store feature metadata (JSON).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation split.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("time", "random"),
        default="random",
        help="Validation split strategy. `time` uses chronological split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=50,
        help="Number of points in threshold search grid (for F1 tuning).",
    )
    return parser.parse_args(args=cli_args)


def _load_and_preprocess(
    train_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    raw_df = pd.read_csv(train_path)
    timestamps = pd.to_datetime(raw_df["transaction_time"], errors="coerce")
    processed = build_features(raw_df)
    y = processed.pop(TARGET_COLUMN).astype(int)
    X = processed
    return X, y, timestamps


def _cat_feature_indices(columns: List[str]) -> List[int]:
    return [columns.index(col) for col in CATEGORICAL_FEATURES if col in columns]


def _train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    strategy: str,
    validation_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if strategy == "random":
        return train_test_split(
            X,
            y,
            test_size=validation_size,
            random_state=random_state,
            stratify=y,
        )

    if timestamps.isna().all():
        raise ValueError(
            "Transaction timestamps are not available; cannot perform time-based split."
        )

    order = timestamps.sort_values().index
    cutoff_idx = int(len(order) * (1 - validation_size))
    cutoff_idx = max(1, min(cutoff_idx, len(order) - 1))
    cutoff_time = timestamps.loc[order[cutoff_idx]]

    mask = timestamps < cutoff_time

    X_train, X_val = X[mask], X[~mask]
    y_train, y_val = y[mask], y[~mask]

    if X_val.empty or X_train.empty:
        raise ValueError(
            "Time-based split produced empty train or validation set. "
            "Try adjusting --validation-size."
        )

    return X_train, X_val, y_train, y_val


def _evaluate(
    model: CatBoostClassifier, X_val: pd.DataFrame, y_val: pd.Series
) -> Dict[str, float]:
    proba = model.predict_proba(X_val)[:, 1]
    predictions = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_val, proba)),
        "f1": float(f1_score(y_val, predictions)),
    }


def _find_best_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    grid_size: int,
) -> Tuple[float, float]:
    grid = np.linspace(0.05, 0.95, grid_size)
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in grid:
        preds = (proba >= threshold).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, best_f1


def train_model(
    train_path: Path,
    model_path: Path,
    metrics_path: Path,
    metadata_path: Path,
    validation_size: float,
    random_state: int,
    split_strategy: str,
    threshold_grid_size: int,
) -> Dict[str, float]:
    X, y, timestamps = _load_and_preprocess(train_path)
    cat_indices = _cat_feature_indices(list(X.columns))

    X_train, X_val, y_train, y_val = _train_val_split(
        X,
        y,
        timestamps,
        strategy=split_strategy,
        validation_size=validation_size,
        random_state=random_state,
    )

    positive = y_train.sum()
    negative = len(y_train) - positive
    scale_pos_weight = negative / max(positive, 1)

    base_params = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        custom_metric=["AUC", "F1", "Logloss"],
        random_seed=random_state,
        scale_pos_weight=scale_pos_weight,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        use_best_model=True,
    )

    best_score = -np.inf
    best_model: Optional[CatBoostClassifier] = None
    best_params: Dict[str, float] = {}

    for params in DEFAULT_PARAM_GRID:
        model = CatBoostClassifier(**base_params, **params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_indices,
            eval_set=(X_val, y_val),
            verbose=False,
        )
        metrics = _evaluate(model, X_val, y_val)
        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model = model
            best_params = params.copy()

    if best_model is None:
        raise RuntimeError(
            "Failed to train CatBoost model with provided parameter grid."
        )

    best_iteration = best_model.get_best_iteration()
    if best_iteration is None or best_iteration <= 0:
        best_iteration = best_model.tree_count_

    final_iterations = max(int(best_iteration), 1)
    final_params = base_params.copy()
    final_params.update(best_params)
    final_params["iterations"] = final_iterations

    final_params["use_best_model"] = False
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(
        X,
        y,
        cat_features=cat_indices,
        verbose=False,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(model_path)

    val_proba = best_model.predict_proba(X_val)[:, 1]
    best_threshold, tuned_f1 = _find_best_threshold(
        y_true=y_val, proba=val_proba, grid_size=threshold_grid_size
    )
    final_metrics = _evaluate(best_model, X_val, y_val)
    final_metrics["f1_best_threshold"] = float(tuned_f1)
    final_metrics["best_threshold"] = float(best_threshold)
    best_scores = best_model.get_best_score()
    best_auc = best_scores.get("validation", {}).get("AUC")
    best_f1 = best_scores.get("validation", {}).get("F1")
    if best_auc is not None:
        final_metrics["catboost_best_auc"] = float(best_auc)
    if best_f1 is not None:
        final_metrics["catboost_best_f1"] = float(best_f1)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "validation": final_metrics,
                "best_iteration": final_iterations,
                "best_params": best_params,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "model_path": str(model_path),
        "decision_threshold": best_threshold,
    }
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(
        f"Training completed. Best ROC-AUC: {final_metrics['roc_auc']:.4f}, "
        f"F1@0.5: {final_metrics['f1']:.4f}, "
        f"F1@best={final_metrics['f1_best_threshold']:.4f} "
        f"(threshold={final_metrics['best_threshold']:.3f}). "
        f"Model saved to {model_path}."
    )
    return final_metrics


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    train_model(
        train_path=args.train_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        metadata_path=args.metadata_path,
        validation_size=args.validation_size,
        random_state=args.random_state,
        split_strategy=args.split_strategy,
        threshold_grid_size=args.threshold_grid_size,
    )


if __name__ == "__main__":
    main()
