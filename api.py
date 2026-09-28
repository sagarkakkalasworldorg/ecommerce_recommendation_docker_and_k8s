from fastapi import FastAPI
import joblib
import pandas as pd


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="E-Commerce Recommendation API",
    description="ML model serving API",
    version="1.0"
)


# ==================================================
# LOAD MODEL
# ==================================================

MODEL_PATH = "/app/model/model.pkl"

model = joblib.load(
    MODEL_PATH
)


print("\n========================================")
print("ML MODEL LOADED")
print("========================================")
print("Model:", MODEL_PATH)
print("========================================\n")


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/modelapi/health")
def health():

    return {
        "status": "healthy",
        "model": "ecommerce-recommendation-model"
    }


# ==================================================
# PREDICTION
# ==================================================

@app.post("/modelapi/predict")
def predict(user_data: dict):

    # ----------------------------------------------
    # Convert incoming JSON to DataFrame
    # ----------------------------------------------

    input_df = pd.DataFrame(
        [user_data]
    )


    # ----------------------------------------------
    # Get expected features from model
    # ----------------------------------------------

    expected_features = list(
        model.feature_names_in_
    )


    # ----------------------------------------------
    # Validate input
    # ----------------------------------------------

    missing_features = [

        feature

        for feature in expected_features

        if feature not in input_df.columns

    ]


    if missing_features:

        return {
            "error": "Missing required features",
            "missing_features": missing_features
        }


    # ----------------------------------------------
    # Keep training feature order
    # ----------------------------------------------

    input_df = input_df[
        expected_features
    ]


    # ----------------------------------------------
    # Make prediction
    # ----------------------------------------------

    prediction = model.predict(
        input_df
    )


    predicted_category = prediction[0]


    # ----------------------------------------------
    # Return prediction
    # ----------------------------------------------

    return {
        "recommended_category": predicted_category
    }