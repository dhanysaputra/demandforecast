import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(
    sales_path: str = None,
    purchase_path: str = None,
):
    """
    Load sales and purchase data from CSV.

    Expected columns (case-insensitive):
      sales_df: DeliveryDate, DeliveredQuantity
      purchase_df: DeliveryDate, DeliveredQuantity, IsConfirmed, RestQuantity, OrderedQuantity
    """
    if sales_path is None:
        sales_path = DATA_DIR / "sample_sales.csv"
    if purchase_path is None:
        purchase_path = DATA_DIR / "sample_purchase.csv"

    sales_df = pd.read_csv(sales_path)
    purchase_df = pd.read_csv(purchase_path)

    # Normalize column names
    sales_df.columns = [c.strip() for c in sales_df.columns]
    purchase_df.columns = [c.strip() for c in purchase_df.columns]

    # Standardize required names
    sales_df = sales_df.rename(
        columns={
            "DeliveryDate__c": "DeliveryDate",
            "DeliveredQuantity__c": "DeliveredQuantity",
        }
    )

    purchase_df = purchase_df.rename(
        columns={
            "DeliveryDate__c": "DeliveryDate",
            "DeliveredQuantity__c": "DeliveredQuantity",
            "IsConfirmed__c": "IsConfirmed",
            "RestQuantity__c": "RestQuantity",
            "OrderedQuantity__c": "OrderedQuantity",
        }
    )

    sales_df["DeliveryDate"] = pd.to_datetime(sales_df["DeliveryDate"])
    purchase_df["DeliveryDate"] = pd.to_datetime(purchase_df["DeliveryDate"])

    return sales_df, purchase_df
