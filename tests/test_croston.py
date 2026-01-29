import pandas as pd
from src.forecasting.croston import croston_sba


def test_croston_sba_basic():
    idx = pd.date_range("2023-01-31", periods=6, freq="ME")
    ts = pd.Series([0, 5, 0, 0, 10, 0], index=idx)

    fitted, future = croston_sba(ts, alpha=0.1, h=3)

    assert len(fitted) == 6
    assert len(future) == 3
    assert (future >= 0).all()
