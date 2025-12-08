import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..feature_engineering import make_time_features


def train_xgb(
    demand_ts: pd.Series,
    model_kwargs: dict = None,
    lags=(1, 2, 3),
    roll_windows=(3, 6),
):
    """
    Train XGBoost baseline forecaster on monthly demand.
    """
    if model_kwargs is None:
        model_kwargs = dict(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )

    df_model = make_time_features(demand_ts, lags=lags, roll_windows=roll_windows)
    features = [c for c in df_model.columns if c not in ("date", "y")]

    model = XGBRegressor(**model_kwargs)
    model.fit(df_model[features], df_model["y"])

    return model, df_model, features


def forecast_xgb(model, df_model: pd.DataFrame, features: list, steps: int = 6):
    """
    Autoregressive rollout for XGB models using last known features.
    Returns Series indexed by future month end.
    """
    last_row = df_model.iloc[-1].copy()

    t = int(last_row["t"])
    current_date = last_row["date"]

    history = list(df_model["y"].tail(6))

    lag1 = float(last_row["lag1"])
    lag2 = float(last_row["lag2"])
    lag3 = float(last_row["lag3"])

    roll_cols = [c for c in df_model.columns if c.startswith("rolling_")]
    roll_state = {c: last_row[c] for c in roll_cols}

    trend = last_row["trend"]

    future_rows = []

    for _ in range(steps):
        current_date = current_date + pd.offsets.MonthEnd(1)
        t += 1
        trend += 1

        row_feat = {
            "t": t,
            "month": current_date.month,
            "lag1": lag1,
            "lag2": lag2,
            "lag3": lag3,
            "trend": trend,
            **roll_state,
        }

        row_df = pd.DataFrame([[row_feat[f] for f in features]], columns=features)
        y_pred = float(model.predict(row_df)[0])
        future_rows.append({"date": current_date, "y_pred_xgb": y_pred})

        history.append(y_pred)
        if len(history) > 6:
            history = history[-6:]

        last3 = history[-3:]
        roll_state["rolling_mean_3"] = np.mean(last3)
        roll_state["rolling_std_3"] = np.std(last3)
        roll_state["rolling_min_3"] = np.min(last3)
        roll_state["rolling_max_3"] = np.max(last3)
        roll_state["rolling_mean_6"] = np.mean(history)
        roll_state["rolling_std_6"] = np.std(history)

        lag3 = lag2
        lag2 = lag1
        lag1 = y_pred

    future_df = pd.DataFrame(future_rows).set_index("date")
    return future_df["y_pred_xgb"]
