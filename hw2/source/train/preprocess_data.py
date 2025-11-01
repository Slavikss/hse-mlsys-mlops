import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

TARGET_COLUMN = "target"

CATEGORICAL_FEATURES: List[str] = [
    "merch",
    "cat_id",
    "gender",
    "us_state",
    "jobs",
    "transaction_dayofweek",
    "transaction_month",
    "post_code",
]

NUMERIC_FEATURES: List[str] = [
    "amount",
    "amount_log",
    "population_city",
    "population_log",
    "transaction_hour",
    "is_weekend",
    "is_night",
    "customer_merchant_distance_km",
    "lat",
    "lon",
    "merchant_lat",
    "merchant_lon",
    "lat_diff_abs",
    "lon_diff_abs",
    "same_location_5km",
]

FEATURE_COLUMNS: List[str] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

WEEKEND_DAYS = {"Saturday", "Sunday"}
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}
EARTH_RADIUS_KM = 6371.0


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _haversine_distance(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    has_target = TARGET_COLUMN in data.columns

    if has_target:
        target = data.pop(TARGET_COLUMN)
    else:
        target = None

    transaction_time = _ensure_datetime(data["transaction_time"])
    data["transaction_hour"] = transaction_time.dt.hour.fillna(-1).astype(int)
    data["transaction_dayofweek"] = transaction_time.dt.day_name().fillna("Unknown")
    data["transaction_month"] = (
        transaction_time.dt.to_period("M").astype(str).fillna("Unknown")
    )
    data["is_weekend"] = data["transaction_dayofweek"].isin(WEEKEND_DAYS).astype(int)
    data["is_night"] = data["transaction_hour"].isin(NIGHT_HOURS).astype(int)

    data["amount"] = data["amount"].fillna(0.0)
    data["amount_log"] = np.log1p(data["amount"])

    data["population_city"] = data["population_city"].fillna(0)
    data["population_log"] = np.log1p(data["population_city"])

    data["post_code"] = data["post_code"].fillna(-1).astype(int).astype(str)

    distance = _haversine_distance(
        data["lat"].astype(float),
        data["lon"].astype(float),
        data["merchant_lat"].astype(float),
        data["merchant_lon"].astype(float),
    )
    data["customer_merchant_distance_km"] = distance.fillna(distance.median())
    data["same_location_5km"] = (data["customer_merchant_distance_km"] <= 5).astype(int)
    data["lat_diff_abs"] = (data["lat"] - data["merchant_lat"]).abs().fillna(0.0)
    data["lon_diff_abs"] = (data["lon"] - data["merchant_lon"]).abs().fillna(0.0)

    for cat_col in CATEGORICAL_FEATURES:
        data[cat_col] = data[cat_col].fillna("unknown").astype(str)

    for num_col in NUMERIC_FEATURES:
        data[num_col] = data[num_col].fillna(0.0)

    data["transaction_hour"] = data["transaction_hour"].astype(int)
    data["is_weekend"] = data["is_weekend"].astype(int)
    data["is_night"] = data["is_night"].astype(int)
    data["same_location_5km"] = data["same_location_5km"].astype(int)

    feature_df = data[FEATURE_COLUMNS].copy()

    if has_target and target is not None:
        feature_df[TARGET_COLUMN] = target

    return feature_df


def preprocess_file(
    input_path: Path, output_path: Optional[Path] = None
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    processed = build_features(df)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".parquet":
            try:
                processed.to_parquet(output_path, index=False)
            except ImportError:
                fallback_path = output_path.with_suffix(".csv")
                processed.to_csv(fallback_path, index=False)
                print(
                    "pyarrow/fastparquet not available; saved CSV instead at "
                    f"{fallback_path}"
                )
        else:
            processed.to_csv(output_path, index=False)

    return processed


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess fraud dataset for CatBoost."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to raw CSV file (train or test).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=False,
        help="Optional path to save preprocessed data (parquet).",
    )
    return parser.parse_args(args=args)


def main(cli_args: Optional[Iterable[str]] = None) -> None:
    args = parse_args(cli_args)
    processed = preprocess_file(args.input_path, args.output_path)
    print(
        f"Processed shape: {processed.shape}. "
        f"Columns: {', '.join(processed.columns.tolist())}"
    )


if __name__ == "__main__":
    main()
