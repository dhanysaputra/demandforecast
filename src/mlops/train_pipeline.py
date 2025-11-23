import pandas as pd
import mlflow
import mlflow.sklearn

from src.data_loader import load_data
from src.inventory.demand_reconstruction import reconstruct_demand
from src.forecasting.hybrid_forecast import hybrid_forecast
from src.inventory.safety_stock import compute_safety_stock
from src.inventory.inventory_simulation import simulate_inventory_with_rop
from src.mlops.evaluate import evaluate_forecast, drift_check, promote_metrics
from src.mlops.persist import save_model, save_series, save_dataframe, save_metrics
from src.mlops.mlflow_utils import (
    start_mlflow_run, log_params, log_metrics, log_artifact_dataframe,
    log_artifact_series, log_artifact_file, register_model
)

def run_training_pipeline(
    model_name="charger_demand_forecast",
    steps=6,
    lead_time_days=7,
    tolerance_early_days=2,
    tolerance_late_days=1,
):

    # -------------------------------
    # 1) MLflow run start
    # -------------------------------
    run = start_mlflow_run(experiment_name="DemandForecast", run_name="NightlyRetrain")
    run_id = run.info.run_id
    print("MLflow run_id:", run_id)

    # -------------------------------
    # 2) Load data
    # -------------------------------
    sales_df, purchase_df = load_data()

    # -------------------------------
    # 3) Reconstruct demand
    # -------------------------------
    inv_df = reconstruct_demand(sales_df, purchase_df)
    demand_ts = inv_df["true_demand_est"].asfreq("ME")

    # -------------------------------
    # 4) Forecast (hybrid model)
    # -------------------------------
    future_forecast, debug = hybrid_forecast(demand_ts, steps=steps)
    xgb_model = debug["xgb_model"]

    # -------------------------------
    # 5) Metrics
    # -------------------------------
    backtest_n = 3
    y_true = demand_ts.tail(backtest_n)
    y_pred = pd.Series([demand_ts.mean()] * backtest_n, index=y_true.index)
    metrics = evaluate_forecast(y_true, y_pred)
    drift = drift_check(metrics)

    full_metrics = {
        **metrics,
        **drift,
        "w": debug["w"],
        "ADI": debug["intermittency"]["ADI"],
        "CV2": debug["intermittency"]["CV2"],
        "class": debug["intermittency"]["class"],
    }

    # -------------------------------
    # 6) Safety Stock
    # -------------------------------
    ss = compute_safety_stock(inv_df, lead_time_days, tolerance_early_days, tolerance_late_days)

    # -------------------------------
    # 7) Inventory simulation
    # -------------------------------
    sim_df = simulate_inventory_with_rop(inv_df, future_forecast, ss, lead_time_days)

    # -------------------------------
    # 8) Save artifacts locally & MLflow
    # -------------------------------
    save_model(xgb_model, "xgb_model.pkl")
    save_series(future_forecast, "forecast.csv")
    save_dataframe(sim_df, "simulation.csv")
    save_metrics(full_metrics, "metrics.json")

    # MLflow logging
    log_params({
        "steps": steps,
        "lead_time_days": lead_time_days,
        "tolerance_early_days": tolerance_early_days,
        "tolerance_late_days": tolerance_late_days,
        "model_name": model_name
    })
    log_metrics(full_metrics)

    log_artifact_dataframe(inv_df, "inventory_history")
    log_artifact_series(future_forecast, "forecast_future")
    log_artifact_dataframe(sim_df, "inventory_sim_future")
    log_artifact_file("artifacts/xgb_model.pkl")

    # -------------------------------
    # 9) MLflow Model Registry
    # -------------------------------
    mlflow.sklearn.log_model(xgb_model, artifact_path="model")
    registered = register_model(model_name, run_id)

    # -------------------------------
    # 10) Promote previous metrics
    # -------------------------------
    promote_metrics()

    mlflow.end_run()
    return registered
