from forecasting.hybrid_forecast import hybrid_forecast
from inventory.demand_reconstruction import reconstruct_demand
from inventory.safety_stock import compute_safety_stock
from inventory.inventory_simulation import simulate_inventory_with_rop
from utils.plots import plot_history_and_forecast
from data_loader import load_data


def main():
    sales_df, purchase_df = load_data()

    inv_df = reconstruct_demand(sales_df, purchase_df)
    demand_ts = inv_df["true_demand_est"].asfreq("ME")

    forecast_future, debug = hybrid_forecast(demand_ts, abc_class="A", steps=6, alpha=0.1)

    ss = compute_safety_stock(inv_df, lead_time_days=7, tolerance_early_days=2, tolerance_late_days=1)

    sim_df = simulate_inventory_with_rop(
        history_df=inv_df,
        forecast_series=forecast_future,
        safety_stock_units=ss,
        lead_time_days=7,
    )

    print("Intermittency:", debug["intermittency"], "w=", debug["w"])
    print(sim_df)

    plot_history_and_forecast(
        inv_df,
        forecast_future,
        safety_stock=ss,
        sim_df=sim_df,
        title="Hybrid Demand Forecast + Inventory Simulation"
    )


if __name__ == "__main__":
    main()
