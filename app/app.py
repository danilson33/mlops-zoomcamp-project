import logging
import os
import warnings
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

EXPERIMENT_NAME = "diamond-default"
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

warnings.filterwarnings("ignore")
logger = logging.getLogger("Diamond Price Prediction API")
logger.setLevel(logging.INFO)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:4566"
)


class Diamond(BaseModel):
    carat: float = None
    cut: str = None
    color: str = None
    clarity: str = None
    depth: float = None
    table: int = None
    x: float = None
    y: float = None
    z: float = None


ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading the model...")

    runs = mlflow.search_runs(experiment_ids=["1"])
    best_run = runs.sort_values("metrics.RMSE", ascending=False).iloc[0]
    best_run_id = best_run["run_id"]
    logger.info(f"Best runL {best_run_id}")

    logged_model = f"runs:/{best_run_id}/sklearn-model"
    ml_model["predict"] = mlflow.pyfunc.load_model(logged_model)
    yield
    # Clean up the ML models and release the resources
    ml_model.clear()


app = FastAPI(lifespan=lifespan, title="Diamond Price Prediction API", version="0.0.1")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Diamond Price Prediction API"}


@app.post("/predict", tags=["prediction"])
async def predict(diamond: Diamond):
    logger.info("Predicting the Diamond Price")
    diamond_dict = diamond.dict()
    diamond_df = pd.DataFrame([diamond_dict])
    categorical_columns = ["cut", "color", "clarity"]
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        diamond_df[col] = label_encoder.fit_transform(diamond_df[col])
    prediction = ml_model["predict"].predict(diamond_df)
    logger.info(prediction[0])

    return {"prediction": prediction[0].item()}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}
