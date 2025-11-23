import numpy as np
import pandas as pd


def compute_safety_stock(
    inv_df: pd.DataFrame,
    lead_time_days: float,
    tolerance_early_days: float = 0.0,
    tolerance_late_days: float = 0.0,
    z: float = 2.05,
):
    """
    Safety stock:
      SS = Z * sqrt( sigma_d^2 * LT + mu_d^2 * sigma_LT^2 )
    """
    demand = inv_df["true_demand_est"].astype(float)

    mu_d = demand.mean()
    sigma_d = demand.std()

    tol_span = tolerance_early_days + tolerance_late_days
    sigma_lt = (tol_span / np.sqrt(12)) if tol_span > 0 else 0.0

    ss = z * np.sqrt((sigma_d ** 2) * lead_time_days + (mu_d ** 2) * (sigma_lt ** 2))
    return float(ss)
