def reorder_point(forecast_demand_per_period: float, lead_time_periods: int, safety_stock_units: float):
    """
    ROP = demand during lead time + safety stock
    """
    return forecast_demand_per_period * lead_time_periods + safety_stock_units
