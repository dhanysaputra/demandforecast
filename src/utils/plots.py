import matplotlib.pyplot as plt

def plot_history_and_forecast(inv_df, forecast_series, safety_stock=None, sim_df=None, title="Demand Forecast"):
    plt.figure(figsize=(16, 9))

    plt.plot(inv_df.index, inv_df["sales_observed"], label="Sales (Observed)", linewidth=2)
    plt.plot(inv_df.index, inv_df["purchase"], label="Purchases", linewidth=2)
    plt.plot(inv_df.index, inv_df["inv_end"], label="Inventory (EOM)", linewidth=2)

    plt.plot(forecast_series.index, forecast_series.values, label="Forecast", linewidth=2)

    if safety_stock is not None:
        plt.plot(inv_df.index, [safety_stock]*len(inv_df), label="Safety Stock", linestyle=":", linewidth=2)

    if sim_df is not None:
        plt.plot(sim_df.index, sim_df["inv_end"], label="Inventory (Simulated)", linestyle="--", linewidth=2)
        plt.plot(sim_df.index, sim_df["recommended_order"], label="Recommended Orders", linewidth=2)
        plt.plot(sim_df.index, sim_df["ROP"], label="Reorder Point", linestyle="--", linewidth=2)

    plt.title(title, fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Units", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
