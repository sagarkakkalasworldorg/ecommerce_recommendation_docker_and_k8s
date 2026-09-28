from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd


# ==================================================
# MLflow configuration
# ==================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"

MODEL_NAME = "ecommerce-recommendation-model"

MODEL_ALIAS = "production"


# ==================================================
# Connect to MLflow
# ==================================================

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ==================================================
# Load production model
# ==================================================

model_uri = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


model = mlflow.sklearn.load_model(
    model_uri
)


print("\n================================")
print("MLflow model loaded")
print("Model:", model_uri)
print("================================\n")


# ==================================================
# Create FastAPI application
# ==================================================

app = FastAPI(

    title="E-Commerce Recommendation API",

    description="MLflow model served using FastAPI",

    version="1.0"
)


# ==================================================
# Request Model
# ==================================================

class PredictionRequest(BaseModel):

    user_data: dict


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "model": model_uri

    }


# ==================================================
# PREDICTION API
# ==================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    # ----------------------------------------------
    # Extract user data
    # ----------------------------------------------

    user_data = request.user_data


    # ----------------------------------------------
    # Convert to DataFrame
    # ----------------------------------------------

    df = pd.DataFrame(
        [user_data]
    )


    # ----------------------------------------------
    # Make prediction
    # ----------------------------------------------

    prediction = model.predict(
        df
    )


    # ----------------------------------------------
    # Return result
    # ----------------------------------------------

    return {

        "recommended_category":
            prediction[0]

    }

