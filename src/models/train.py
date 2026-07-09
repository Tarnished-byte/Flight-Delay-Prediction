from __future__ import annotations
import os
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, classification_report,
)
from xgboost import XGBClassifier
from src.features.build_features import prepare_training_data, compute_smoothed_target_encoding

DATA_PATH = "data/processed/flights_sample_3m_cleaned.csv"
MODEL_PATH = "models/xgb_flight_delay.joblib"


def get_scale_pos_weight(y: pd.Series) -> float:
    neg, pos = (y == 0).sum(), (y == 1).sum()
    return neg / pos


def carve_validation_set(train_df, X_train, y_train, val_fraction: float = 0.1):
    n_val = int(len(train_df) * val_fraction)
    fit_end = len(train_df) - n_val
    return (X_train[:fit_end], X_train[fit_end:],
            y_train.iloc[:fit_end], y_train.iloc[fit_end:])


def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    (X_train, X_test, y_train, y_test,
     preprocessor, feature_names, train_df, test_df) = prepare_training_data(df)

    X_fit, X_val, y_fit, y_val = carve_validation_set(train_df, X_train, y_train)

    params = {
        "n_estimators": 600,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "scale_pos_weight": get_scale_pos_weight(y_fit),
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "n_jobs": -1,
        "random_state": 42,
    }

    mlflow.set_experiment("flight-delay-prediction")
    with mlflow.start_run():
        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        metrics = {
            "roc_auc": roc_auc_score(y_test, probs),
            "pr_auc": average_precision_score(y_test, probs),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(classification_report(y_test, preds))
        print(metrics)

        importances = pd.Series(model.feature_importances_, index=feature_names)
        print("\nTop 15 features:\n", importances.sort_values(ascending=False).head(15))

        # --- error analysis (moved here, right after preds/probs exist) ---
        test_df = test_df.copy()
        test_df["pred"] = preds
        test_df["pred_prob"] = probs
        wrong = test_df[test_df["pred"] != test_df["is_delayed"]]
        error_rate_by_period = wrong.groupby("departure_period").size() / test_df.groupby("departure_period").size()
        print(error_rate_by_period.sort_values(ascending=False))

        error_rate_by_airline = wrong.groupby("AIRLINE").size() / test_df.groupby("AIRLINE").size()
        print(error_rate_by_airline.sort_values(ascending=False).head(10))
                # --- save rate maps for serving (now model exists, correctly placed) ---
        rate_maps = {}
        for col, new_col in [("AIRLINE", "carrier_delay_rate"), ("ORIGIN", "origin_delay_rate"),
                              ("DEST", "dest_delay_rate"), ("route", "route_delay_rate")]:
            rate_maps[new_col] = compute_smoothed_target_encoding(train_df, col, "is_delayed")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            "model": model,
            "preprocessor": preprocessor,
            "rate_maps": rate_maps,
            "global_mean": train_df["is_delayed"].mean(),
        }, MODEL_PATH)

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()