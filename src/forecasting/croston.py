import numpy as np
import pandas as pd


def croston_sba(ts: pd.Series, alpha: float = 0.1, h: int = 6):
    """
    Croston's method with SBA correction for intermittent demand.
    Returns:
      fitted_sba (Series aligned to ts)
      future_sba (Series length h)
    """
    y = ts.values.astype(float)
    n = len(y)

    if np.all(y == 0):
        fitted = pd.Series(np.zeros(n), index=ts.index)
        future_idx = pd.date_range(
            ts.index[-1] + pd.offsets.MonthEnd(1),
            periods=h,
            freq="ME",
        )
        future = pd.Series(np.zeros(h), index=future_idx)
        return fitted, future

    q = np.zeros(n)
    a = np.zeros(n)
    f = np.zeros(n)

    first = np.argmax(y > 0)
    q[first] = y[first]
    a[first] = 1.0
    f[first] = q[first] / a[first]
    last_demand_idx = first

    for t in range(first + 1, n):
        if y[t] > 0:
            interval = t - last_demand_idx
            q[t] = alpha * y[t] + (1 - alpha) * q[last_demand_idx]
            a[t] = alpha * interval + (1 - alpha) * a[last_demand_idx]
            last_demand_idx = t
        else:
            q[t] = q[last_demand_idx]
            a[t] = a[last_demand_idx]

        f[t] = (q[t] / a[t]) if a[t] > 0 else 0.0

    fitted_sba = (1 - alpha / 2) * f
    future_idx = pd.date_range(
        ts.index[-1] + pd.offsets.MonthEnd(1),
        periods=h,
        freq="ME",
    )
    future_sba = np.repeat(fitted_sba[-1], h)

    return pd.Series(fitted_sba, index=ts.index), pd.Series(future_sba, index=future_idx)
