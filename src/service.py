import bentoml
import mlflow
import pandas as pd
from src.recommend_products import recommend_products #added  for recommendation model
import os #for docker image


# ==================================================
# MLflow CONFIGURATION
# ==================================================

MLFLOW_TRACKING_URI = "http://localhost:5000"

MODEL_NAME = "ecommerce-recommendation-model"

MODEL_ALIAS = "production"


# ==================================================
# CONNECT TO MLFLOW | uncomment before using docker image
# ==================================================



# ==================================================
# CONNECT TO MLFLOW | uncomment after using docker image
# ==================================================

# MLFLOW_TRACKING_URI = os.getenv(
#     "MLFLOW_TRACKING_URI",
#     "http://mlflow:5000"
# )

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)



# ==================================================
# LOAD PRODUCTION MODEL
# ==================================================

model_uri = (
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
)


model = mlflow.sklearn.load_model(
    model_uri
)


# ==================================================
# GET EXPECTED FEATURES
# ==================================================

expected_features = list(
    model.feature_names_in_
)


print("\n========================================")
print("MLflow model loaded")
print("========================================")

print(
    "Model:",
    model_uri
)

print(
    "\nExpected features:"
)

for feature in expected_features:

    print(
        "-",
        feature
    )

print("========================================\n")


# ==================================================
# BENTOML SERVICE
# ==================================================

@bentoml.service
class RecommendationService:


    # ==================================================
    # HEALTH CHECK
    # ==================================================

    @bentoml.api
    def health(self) -> dict:

        return {

            "status": "healthy",

            "model": model_uri

        }


    # ==================================================
    # PREDICTION
    # ==================================================

    @bentoml.api
    def predict(
        self,
        user_data: dict
    ) -> dict:


        # ----------------------------------------------
        # Convert JSON to DataFrame
        # ----------------------------------------------

        input_df = pd.DataFrame(
            [user_data]
        )


        # ----------------------------------------------
        # Validate required features
        # ----------------------------------------------

        missing_features = [

            feature

            for feature in expected_features

            if feature not in input_df.columns

        ]


        if missing_features:

            return {

                "error":
                    "Missing required features",

                "missing_features":
                    missing_features

            }


        # ----------------------------------------------
        # Keep exact training feature order
        # ----------------------------------------------

        input_df = input_df[
            expected_features
        ]


        # ----------------------------------------------
        # Make prediction | added for recommendation model
        # ----------------------------------------------

        prediction = model.predict(
            input_df
        )

        predicted_category = prediction[0]


        recommended_products = recommend_products(
            predicted_category,
            number_of_products=5
        )


        return {

            "recommended_category":
                predicted_category,

            "products":
                recommended_products
        }


        # ----------------------------------------------
        # Return result | comment after adding recommend_product.py
        # ----------------------------------------------

        # return {

        #     "recommended_category":
        #         prediction[0]

        # }
    
        
