import pandas as pd
from src.inventory.safety_stock import compute_safety_stock


def test_safety_stock():
    idx = pd.date_range("2023-01-31", periods=6, freq="ME")
    inv_df = pd.DataFrame({"true_demand_est": [10, 15, 12, 20, 17, 19]}, index=idx)

    ss = compute_safety_stock(
        inv_df, lead_time_days=7, tolerance_early_days=2, tolerance_late_days=1
    )

    assert ss > 0
    assert ss < 1000
