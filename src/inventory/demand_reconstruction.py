import pandas as pd
import numpy as np


def reconstruct_demand(
    sales_df: pd.DataFrame,
    purchase_df: pd.DataFrame,
    freq: str = "M",
    confirmed_only: bool = True,
):
    """
    Build inv_df with:
      purchase, sales_observed, inv_start, sales_served, lost_sales_est,
      inv_end, stockout_flag, true_demand_est
    """
    df_p = purchase_df.copy()

    if confirmed_only and "IsConfirmed" in df_p.columns:
        df_p = df_p[df_p["IsConfirmed"] == True]

    if "RestQuantity" in df_p.columns and "OrderedQuantity" in df_p.columns:
        df_p = df_p[df_p["RestQuantity"] < df_p["OrderedQuantity"]]

    purchase_ts = (
        df_p.groupby(pd.Grouper(key="DeliveryDate", freq=freq))["DeliveredQuantity"]
        .sum()
        .fillna(0)
    )
    sales_ts = (
        sales_df.groupby(pd.Grouper(key="DeliveryDate", freq=freq))["DeliveredQuantity"]
        .sum()
        .asfreq(freq, fill_value=0)
    )

    all_idx = sales_ts.index.union(purchase_ts.index)
    df_inv = pd.DataFrame(index=all_idx)
    df_inv["sales"] = sales_ts.reindex(all_idx).fillna(0)
    df_inv["purchase"] = purchase_ts.reindex(all_idx).fillna(0)

    inv = 0.0
    records = []

    for date, row in df_inv.iterrows():
        purch = float(row["purchase"])
        demand_obs = float(row["sales"])

        inv_start = inv + purch

        if inv_start >= demand_obs:
            sales_served = demand_obs
            lost_sales = 0.0
            inv_end = inv_start - sales_served
            stockout_flag = False
        else:
            sales_served = inv_start
            lost_sales = demand_obs - inv_start
            inv_end = 0.0
            stockout_flag = True

        true_demand_est = sales_served + lost_sales

        records.append(
            {
                "date": date,
                "purchase": purch,
                "sales_observed": demand_obs,
                "inv_start": inv_start,
                "sales_served": sales_served,
                "lost_sales_est": lost_sales,
                "inv_end": inv_end,
                "stockout_flag": stockout_flag,
                "true_demand_est": true_demand_est,
            }
        )
        inv = inv_end

    return pd.DataFrame(records).set_index("date")
