import pandas as pd
from src.forecasting.xgb_model import train_xgb, forecast_xgb

def test_xgb_training_and_forecast():
    # Synthetic monthly demand
    idx = pd.date_range("2023-01-31", periods=12, freq="ME")
    demand_ts = pd.Series([10,12,15,13,14,16,18,17,19,21,20,22], index=idx)

    model, df_model, features = train_xgb(demand_ts)
    assert model is not None
    assert len(features) > 0
    assert "lag1" in features

    future = forecast_xgb(model, df_model, features, steps=3)
    assert len(future) == 3
    assert not future.isna().any()
