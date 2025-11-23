import pandas as pd
from pathlib import Path

from src.utils.metrics import mape, mae


ARTIFACT_DIR = Path("artifacts")


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series):
    y_true, y_pred = y_true.align(y_pred, join="inner")
    return {
        "MAE": float(mae(y_true, y_pred)),
        "MAPE": float(mape(y_true, y_pred)),
        "n_points": int(len(y_true)),
    }


def drift_check(new_metrics: dict, tolerance_pct: float = 25.0):
    """
    If previous metrics exist, compare and flag large degradation.
    """
    prev_path = ARTIFACT_DIR / "metrics_prev.json"
    if not prev_path.exists():
        return {"drift_flag": False, "reason": "no previous metrics"}

    import json

    prev = json.loads(prev_path.read_text())

    drift_flag = False
    reasons = []
    for k in ["MAE", "MAPE"]:
        if k in prev and k in new_metrics:
            if prev[k] == 0:
                continue
            change_pct = (new_metrics[k] - prev[k]) / prev[k] * 100
            if change_pct > tolerance_pct:
                drift_flag = True
                reasons.append(f"{k} worsened by {change_pct:.1f}%")

    return {"drift_flag": drift_flag, "reason": "; ".join(reasons) or "ok"}


def promote_metrics():
    """
    After a successful run, store current metrics as previous baseline.
    """
    cur = ARTIFACT_DIR / "metrics.json"
    prev = ARTIFACT_DIR / "metrics_prev.json"
    if cur.exists():
        prev.write_text(cur.read_text())
