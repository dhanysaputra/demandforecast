from pathlib import Path
import joblib
import pandas as pd


ARTIFACT_DIR = Path("artifacts")


def ensure_dir():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def save_model(model, name="xgb_model.pkl"):
    ensure_dir()
    joblib.dump(model, ARTIFACT_DIR / name)


def save_series(series: pd.Series, name="forecast.csv"):
    ensure_dir()
    series.to_csv(ARTIFACT_DIR / name, header=True)


def save_dataframe(df: pd.DataFrame, name="simulation.csv"):
    ensure_dir()
    df.to_csv(ARTIFACT_DIR / name)


def save_metrics(metrics: dict, name="metrics.json"):
    ensure_dir()
    import json

    with open(ARTIFACT_DIR / name, "w") as f:
        json.dump(metrics, f, indent=2)
