import pandas as pd
from src.inventory.demand_reconstruction import reconstruct_demand


def test_demand_reconstruction():
    sales_df = pd.DataFrame(
        {
            "DeliveryDate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "DeliveredQuantity": [10, 20, 30],
        }
    )

    purchase_df = pd.DataFrame(
        {
            "DeliveryDate": ["2023-01-31", "2023-02-28", "2023-03-31"],
            "DeliveredQuantity": [50, 0, 20],
            "IsConfirmed": [True, True, True],
            "RestQuantity": [0, 0, 0],
            "OrderedQuantity": [50, 0, 20],
        }
    )

    sales_df["DeliveryDate"] = pd.to_datetime(sales_df["DeliveryDate"])
    purchase_df["DeliveryDate"] = pd.to_datetime(purchase_df["DeliveryDate"])

    inv_df = reconstruct_demand(sales_df, purchase_df)

    assert "true_demand_est" in inv_df.columns
    assert len(inv_df) == 3
    assert inv_df["sales_served"].iloc[0] == 10
