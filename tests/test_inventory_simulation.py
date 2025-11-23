import pandas as pd
from src.inventory.inventory_simulation import simulate_inventory_with_rop

def test_inventory_simulation():
    idx_hist = pd.date_range("2023-01-31", periods=4, freq="M")
    history_df = pd.DataFrame({
        "purchase": [20, 0, 10, 0],
        "sales_observed": [10, 15, 5, 8],
        "inv_end": [10, 0, 5, 0]
    }, index=idx_hist)

    idx_future = pd.date_range("2023-05-31", periods=3, freq="M")
    forecast_series = pd.Series([10, 12, 14], index=idx_future)

    sim_df = simulate_inventory_with_rop(
        history_df=history_df,
        forecast_series=forecast_series,
        safety_stock_units=5,
        lead_time_days=7,
    )

    assert len(sim_df) == 3
    assert "recommended_order" in sim_df.columns
    assert (sim_df["inv_end"] >= 0).all()
