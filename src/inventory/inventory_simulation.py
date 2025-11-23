import numpy as np
import pandas as pd
from .reorder_point import reorder_point

def simulate_inventory_with_rop(
    history_df: pd.DataFrame,
    forecast_series: pd.Series,
    safety_stock_units: float,
    lead_time_days: float,
    review_period_days: float = 30.0,
    initial_inventory: float = None,
    lot_size: float = None,
    min_order_qty: float = 0.0,
):
    """
    Monthly inventory simulation with reorder point (ROP).

    forecast_series should be monthly and indexed by month-end.
    """
    lead_time_periods = max(1, int(np.ceil(lead_time_days / review_period_days)))
    idx_future = forecast_series.index

    if initial_inventory is None:
        initial_inventory = float(history_df["inv_end"].iloc[-1]) if "inv_end" in history_df.columns else 0.0

    pipeline = [0.0] * lead_time_periods
    inv = initial_inventory
    rows = []

    for dt in idx_future:
        demand = float(forecast_series.loc[dt])

        receipt = pipeline.pop(0)
        inv_start = inv + receipt

        sales_served = min(inv_start, demand)
        lost_sales = demand - sales_served
        inv_end = inv_start - sales_served

        rop_val = reorder_point(demand, lead_time_periods, safety_stock_units)

        if inv_end <= rop_val:
            target = rop_val
            order_qty = max(0.0, target - inv_end)
            order_qty = max(order_qty, min_order_qty)

            if lot_size is not None and lot_size > 0:
                order_qty = np.ceil(order_qty / lot_size) * lot_size
        else:
            order_qty = 0.0

        pipeline.append(order_qty)

        rows.append({
            "date": dt,
            "forecast_demand": demand,
            "receipt": receipt,
            "inv_start": inv_start,
            "sales_served": sales_served,
            "lost_sales": lost_sales,
            "inv_end": inv_end,
            "ROP": rop_val,
            "recommended_order": order_qty,
            "pipeline_open_orders": sum(pipeline),
        })

        inv = inv_end

    sim_df = pd.DataFrame(rows).set_index("date")
    sim_df["lead_time_periods"] = lead_time_periods
    sim_df["safety_stock_units"] = safety_stock_units
    return sim_df
