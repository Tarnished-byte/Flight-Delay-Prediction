from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REQUIRED_COLUMNS = [
    "FL_DATE", "AIRLINE", "ORIGIN", "ORIGIN_CITY", "DEST", "DEST_CITY",
    "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME", "DISTANCE", "is_delayed",
]

NUMERIC_FEATURES = [
    "DISTANCE", "CRS_ELAPSED_TIME", "sched_hour", "month", "day_of_week",
    "is_weekend", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "dow_sin", "dow_cos", "carrier_delay_rate", "origin_delay_rate",
    "dest_delay_rate", "route_delay_rate",
]
CATEGORICAL_FEATURES = ["distance_bucket", "departure_period"]


def validate_dataframe(df: pd.DataFrame, target: str = "is_delayed") -> None:
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    bad_dates = pd.to_datetime(df["FL_DATE"], errors="coerce").isna().sum()
    if bad_dates:
        raise ValueError(f"{bad_dates} rows have invalid FL_DATE values.")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
    df["sched_hour"] = (df["CRS_DEP_TIME"] // 100).astype(int).clip(0, 23)
    df["month"] = df["FL_DATE"].dt.month
    df["day_of_week"] = df["FL_DATE"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos pairs. Kept alongside raw values — see note above on trees vs sin/cos."""
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["sched_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["sched_hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_route_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["route"] = df["ORIGIN"] + "_" + df["DEST"]
    return df


def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance_bucket"] = pd.cut(
        df["DISTANCE"], bins=[0, 500, 1000, 2000, np.inf],
        labels=["short", "medium", "long", "very_long"],
    )
    return df


def add_departure_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [-1, 5, 11, 16, 20, 24]
    labels = ["Night", "Morning", "Afternoon", "Evening", "Night"]
    df["departure_period"] = pd.cut(
        df["sched_hour"], bins=bins, labels=labels, ordered=False
    ).astype(str)
    return df


def engineer_deterministic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_cyclical_features(df)
    df = add_route_feature(df)
    df = add_distance_bucket(df)
    df = add_departure_period(df)
    return df


def time_based_split(df: pd.DataFrame, date_col: str = "FL_DATE", split_date=None):
    """Train = everything before split_date, test = after. Default: last 15% by date."""
    df = df.sort_values(date_col).reset_index(drop=True)
    if split_date is None:
        split_date = df[date_col].iloc[int(len(df) * 0.85)]
    train_df = df[df[date_col] < split_date].copy()
    test_df = df[df[date_col] >= split_date].copy()
    return train_df, test_df


def compute_smoothed_target_encoding(train_df, group_col, target_col, m: float = 50.0):
    """smoothed_mean = (n * cat_mean + m * global_mean) / (n + m) — protects rare categories."""
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(group_col)[target_col].agg(["mean", "count"])
    return (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)


def apply_target_encoding(df, rate_map, group_col, new_col, global_mean):
    df = df.copy()
    df[new_col] = df[group_col].map(rate_map).fillna(global_mean)
    return df


def fit_apply_target_encodings(train_df, test_df, target, m: float = 50.0):
    global_mean = train_df[target].mean()
    encode_map = {"AIRLINE": "carrier_delay_rate", "ORIGIN": "origin_delay_rate",
                  "DEST": "dest_delay_rate", "route": "route_delay_rate"}
    for col, new_col in encode_map.items():
        rate_map = compute_smoothed_target_encoding(train_df, col, target, m=m)
        train_df = apply_target_encoding(train_df, rate_map, col, new_col, global_mean)
        test_df = apply_target_encoding(test_df, rate_map, col, new_col, global_mean)
    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def prepare_training_data(df: pd.DataFrame, target: str = "is_delayed",
                           split_date=None, m: float = 50.0):
    validate_dataframe(df, target)
    df = engineer_deterministic_features(df)

    train_df, test_df = time_based_split(df, split_date=split_date)
    train_df, test_df = fit_apply_target_encodings(train_df, test_df, target, m=m)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    X_test = preprocessor.transform(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    feature_names = preprocessor.get_feature_names_out()

    return (X_train, X_test, train_df[target], test_df[target],
            preprocessor, feature_names, train_df, test_df)


if __name__ == "__main__":
    df = pd.read_csv("ML-P2/data/processed/flights_sample_3m_cleaned.csv")
    X_train, X_test, y_train, y_test, preprocessor, feature_names, train_df, test_df = prepare_training_data(df)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train range: {train_df['FL_DATE'].min()} -> {train_df['FL_DATE'].max()}")
    print(f"Test range:  {test_df['FL_DATE'].min()} -> {test_df['FL_DATE'].max()}")