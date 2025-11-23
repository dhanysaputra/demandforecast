import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path

def start_mlflow_run(experiment_name="demand_forecasting", run_name="nightly_retrain"):
    """
    Starts or creates an MLflow experiment & run.
    """
    mlflow.set_tracking_uri("file:./mlruns")  # local tracking folder
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, float(v))


def log_artifact_file(path: str):
    mlflow.log_artifact(path)


def log_artifact_dataframe(df: pd.DataFrame, name: str):
    temp_path = Path("artifacts") / f"{name}.csv"
    df.to_csv(temp_path)
    mlflow.log_artifact(str(temp_path))


def log_artifact_series(series: pd.Series, name: str):
    temp_path = Path("artifacts") / f"{name}.csv"
    series.to_csv(temp_path, header=True)
    mlflow.log_artifact(str(temp_path))


def register_model(model_name: str, run_id: str, artifact_path="model"):
    """
    Register model in MLflow Model Registry.
    """
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/{artifact_path}",
        name=model_name
    )
    print(f"Registered model version: {result.version}")
    return result
