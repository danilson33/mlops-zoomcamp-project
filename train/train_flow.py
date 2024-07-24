import io
import logging
import os
import warnings
from tempfile import TemporaryDirectory

import boto3
import mlflow.xgboost
import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:4566"
)
EXPERIMENT_NAME = "diamond-default"


def get_s3_client():
    s3_endpoint_url = os.getenv("AWS_S3_ENDPOINT", "http://localhost:4566")
    s3 = boto3.client("s3", endpoint_url=s3_endpoint_url)

    return s3


@task(log_prints=True, retries=3, tags=["read_data"])
def read_data(s3_client, bucket_name: str, file_name: str) -> pd.DataFrame:
    logging.info("Reading data from s3 ...")
    with TemporaryDirectory():
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

    return df


@task(log_prints=True, tags=["preprocess"])
def data_prep(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Data prep ...")
    data = data.drop(["Unnamed: 0"], axis=1)
    data = data.drop(data[data["x"] == 0].index)
    data = data.drop(data[data["y"] == 0].index)
    data = data.drop(data[data["z"] == 0].index)
    data = data[(data["depth"] < 75) & (data["depth"] > 45)]
    data = data[(data["table"] < 80) & (data["table"] > 40)]
    data = data[(data["x"] < 30)]
    data = data[(data["y"] < 30)]
    data = data[(data["z"] < 30) & (data["z"] > 2)]

    return data


@task(log_prints=True, tags=["split_data"])
def data_split(data: pd.DataFrame):
    categorical_columns = ["cut", "color", "clarity"]
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    X = data.drop(["price"], axis=1)
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    return X_train, X_test, y_train, y_test


@task(log_prints=True, tags=["train_model"])
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    ntrees=100,
    max_depth=7,
    lr=0.1,
):
    with mlflow.start_run():
        # Train model
        xgbRegressor = XGBRegressor(
            max_depth=max_depth,
            n_estimators=ntrees,
            learning_rate=lr,
            random_state=42,
            seed=42,
            reg_lambda=1,
            gamma=1,
        )
        pipeline = Pipeline(steps=[("regressor", xgbRegressor)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print(
            "XGBoost tree model (max_depth=%f, trees=%f, lr=%f):"
            % (max_depth, ntrees, lr)
        )
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("ntrees", ntrees)
        mlflow.log_param("lr", lr)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("MAE", mae)

        mlflow.sklearn.log_model(
            sk_model=xgbRegressor,
            artifact_path="sklearn-model",
            # signature=signature,
            registered_model_name="xgboost-model",
        )
        mlflow.end_run()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return (rmse, mae, r2)


@flow(name="training")
def train_flow():
    s3_client = get_s3_client()
    data = read_data(
        s3_client,
        os.getenv("DATA_BUCKET", "data"),
        os.getenv("DATA_FILE", "diamonds.csv"),
    )
    prepared_data = data_prep(data)
    X_train, X_test, y_train, y_test = data_split(prepared_data)
    train_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_flow()
