import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

_DEFAULT_MPL_DIR = Path.cwd() / ".matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPL_DIR))
_DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier

CURRENT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CURRENT_DIR / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from train.preprocess_data import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    build_features,
)


def parse_args(cli_args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CatBoost fraud detection inference pipeline."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing input CSV file.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="test.csv",
        help="Name of the input CSV file inside input directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to store prediction artifacts.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/catboost_model.cbm"),
        help="Path to trained CatBoost model.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("artifacts/feature_metadata.json"),
        help="Path to JSON metadata with feature configuration.",
    )
    parser.add_argument(
        "--submission-filename",
        type=str,
        default="sample_submission.csv",
        help="Filename for prediction CSV (Kaggle submission format).",
    )
    parser.add_argument(
        "--importances-filename",
        type=str,
        default="feature_importances.json",
        help="Filename for top-5 feature importances JSON.",
    )
    parser.add_argument(
        "--density-filename",
        type=str,
        default="prediction_density.png",
        help="Filename for probability density plot.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for converting probabilities to class labels.",
    )
    return parser.parse_args(args=cli_args)


def _load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _prepare_features(
    raw_df: pd.DataFrame, expected_features: List[str]
) -> pd.DataFrame:
    features = build_features(raw_df)
    # Align column order and ensure all expected columns exist
    missing = set(expected_features) - set(features.columns)
    if missing:
        raise ValueError(f"Missing features after preprocessing: {sorted(missing)}")
    extra = set(features.columns) - set(expected_features)
    if extra:
        features = features.drop(columns=extra)
    return features[expected_features]


def _predict(
    model_path: Path,
    features: pd.DataFrame,
) -> np.ndarray:
    model = CatBoostClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(features)[:, 1]
    return probabilities


def _save_submission(
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> pd.DataFrame:
    predictions = (probabilities >= threshold).astype(int)
    submission = pd.DataFrame(
        {
            "index": np.arange(len(predictions)),
            "prediction": predictions,
        }
    )
    submission.to_csv(output_path, index=False)
    return submission


def _save_importances(
    model_path: Path,
    feature_names: List[str],
    output_path: Path,
) -> dict:
    model = CatBoostClassifier()
    model.load_model(model_path)
    importances = model.get_feature_importance(prettified=False)
    pairs = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    top_importances = {name: float(score) for name, score in pairs}
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(top_importances, fp, ensure_ascii=False, indent=2)
    return top_importances


def _save_density_plot(
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    sns.kdeplot(probabilities, fill=True, color="royalblue")
    plt.title("Density of Predicted Fraud Probabilities")
    plt.xlabel("Fraud probability")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)

    input_path = args.input_dir / args.input_file
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(args.metadata_path)
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
    metadata_threshold = metadata.get("decision_threshold")

    raw_df = pd.read_csv(input_path)
    features = _prepare_features(raw_df, feature_columns)
    probabilities = _predict(args.model_path, features)

    threshold = args.threshold if args.threshold is not None else metadata_threshold
    if threshold is None:
        threshold = 0.5

    submission_path = output_dir / args.submission_filename
    submission = _save_submission(probabilities, threshold, submission_path)

    importances_path = output_dir / args.importances_filename
    top_importances = _save_importances(
        args.model_path, feature_columns, importances_path
    )

    density_path = output_dir / args.density_filename
    _save_density_plot(probabilities, density_path)

    print(
        f"Inference completed on {len(submission)} rows.\n"
        f"Submission saved to: {submission_path}\n"
        f"Threshold used: {threshold:.3f}\n"
        f"Top-5 importances: {top_importances}\n"
        f"Density plot saved to: {density_path}"
    )


if __name__ == "__main__":
    main()
