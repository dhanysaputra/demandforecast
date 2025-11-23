import pandas as pd
from src.forecasting.hybrid_forecast import hybrid_forecast

def test_hybrid_forecast_output():
    idx = pd.date_range("2023-01-31", periods=12, freq="M")
    demand_ts = pd.Series([0,5,0,10,0,3,0,12,0,5,0,1], index=idx)

    future, debug = hybrid_forecast(demand_ts, steps=4)

    assert len(future) == 4
    assert "w" in debug
    assert debug["w"] > 0
    assert not future.isna().any()
