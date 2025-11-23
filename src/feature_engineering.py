import numpy as np
import pandas as pd

def make_time_features(demand_ts: pd.Series, lags=(1,2,3), roll_windows=(3,6)):
    """
    Create supervised learning features from a monthly demand series.

    Returns DataFrame with columns:
      date, y, t, month, lagk, rolling_mean_w, rolling_std_w, rolling_min_w, rolling_max_w, trend
    """
    df = demand_ts.to_frame(name="y").reset_index().rename(columns={"index":"date"})
    df["t"] = np.arange(len(df))
    df["month"] = df["date"].dt.month

    for lag in lags:
        df[f"lag{lag}"] = df["y"].shift(lag)

    for w in roll_windows:
        df[f"rolling_mean_{w}"] = df["y"].rolling(w).mean().shift(1)
        df[f"rolling_std_{w}"]  = df["y"].rolling(w).std().shift(1)

    # always include short-window min/max for local spike cues
    df["rolling_min_3"] = df["y"].rolling(3).min().shift(1)
    df["rolling_max_3"] = df["y"].rolling(3).max().shift(1)

    df["trend"] = df["t"]
    return df.dropna().reset_index(drop=True)
