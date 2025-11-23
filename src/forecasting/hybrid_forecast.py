import numpy as np
import pandas as pd
from .xgb_model import train_xgb, forecast_xgb
from .croston import croston_sba

def classify_adi_cv2(ts: pd.Series):
    """
    Syntetos–Boylan intermittency classification.
    Returns dict with ADI, CV2, class in {"X","Y","Z"}.
    """
    y = ts.values.astype(float)
    nz = y[y > 0]

    if len(nz) == 0:
        return {"ADI": np.inf, "CV2": np.inf, "class": "Z"}

    ADI = len(y) / len(nz)
    CV2 = (nz.std() / nz.mean())**2 if nz.mean() > 0 else np.inf

    if ADI < 1.32 and CV2 < 0.49:
        klass = "X"
    elif ADI < 1.32 and CV2 >= 0.49:
        klass = "Y"
    else:
        klass = "Z"

    return {"ADI": ADI, "CV2": CV2, "class": klass}


def automatic_hybrid_weight(ts: pd.Series, abc_class: str = "A"):
    """
    Decide blend weight w for hybrid forecast:
      y_hybrid = w * y_xgb + (1-w) * y_croston
    """
    info = classify_adi_cv2(ts)
    klass = info["class"]

    if klass == "X":
        w = 0.85
    elif klass == "Y":
        w = 0.70
    else:
        w = 0.55

    abc_class = abc_class.upper()
    if abc_class == "A":
        w += 0.05
    elif abc_class == "C":
        w -= 0.05

    return max(0.3, min(0.9, w)), info


def hybrid_forecast(
    demand_ts: pd.Series,
    abc_class: str = "A",
    steps: int = 6,
    alpha: float = 0.1,
):
    """
    Hybrid intermittent-demand forecast.
    Returns (full_pred, debug_dict)
    """
    # XGB baseline
    xgb_model, df_model, features = train_xgb(demand_ts)
    xgb_future = forecast_xgb(xgb_model, df_model, features, steps=steps)

    # Croston SBA
    fitted_sba, future_sba = croston_sba(demand_ts, alpha=alpha, h=steps)

    # weight selection
    w, info = automatic_hybrid_weight(demand_ts, abc_class=abc_class)

    # Align + blend future
    future_idx = xgb_future.index
    sba_future = future_sba.reindex(future_idx).values
    hybrid_future = w * xgb_future.values + (1-w) * sba_future

    hybrid_future = pd.Series(hybrid_future, index=future_idx, name="y_pred_hybrid")

    debug = {
        "w": w,
        "intermittency": info,
        "xgb_future": xgb_future,
        "sba_future": future_sba,
        "xgb_model": xgb_model,
        "df_model": df_model,
        "features": features,
    }

    return hybrid_future, debug
