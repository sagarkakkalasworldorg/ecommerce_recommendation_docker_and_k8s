import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==================================================
# 1. LOAD DATA
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "data/ml_training_data.csv"
)

print(
    "Dataset shape:",
    df.shape
)


# ==================================================
# 2. SEPARATE FEATURES AND TARGET
# ==================================================

X = df.drop(
    [
        "user_id",
        "preferred_category"
    ],
    axis=1
)

y = df[
    "preferred_category"
]


# ==================================================
# 3. CATEGORICAL FEATURES
# ==================================================

categorical_features = [
    "gender",
    "location"
]


# ==================================================
# 4. NUMERICAL FEATURES
# ==================================================

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


# ==================================================
# 5. PREPROCESSOR
# ==================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_features
        ),

        (
            "numerical",

            "passthrough",

            numerical_features
        )
    ]
)


# ==================================================
# 6. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ==================================================
# 7. MLflow TRACKING SERVER
# ==================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)


# ==================================================
# 8. MLflow EXPERIMENT
# ==================================================

mlflow.set_experiment(
    "ecommerce-recommendation"
)


# ==================================================
# 9. MODEL CONFIGURATION
# ==================================================

models = [

    {
        "name": "logistic-regression",

        "model": LogisticRegression(
            max_iter=1000
        ),

        "params": {
            "model_type": "LogisticRegression",
            "max_iter": 1000
        }
    },


    {
        "name": "decision-tree",

        "model": DecisionTreeClassifier(
            max_depth=10,
            random_state=42
        ),

        "params": {
            "model_type": "DecisionTreeClassifier",
            "max_depth": 10,
            "random_state": 42
        }
    },


    {
        "name": "random-forest-50",

        "model": RandomForestClassifier(
            n_estimators=50,
            random_state=42
        ),

        "params": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 50,
            "random_state": 42
        }
    },


    {
        "name": "random-forest-100",

        "model": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "params": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42
        }
    },


    {
        "name": "random-forest-200",

        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "params": {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "random_state": 42
        }
    }
]


# ==================================================
# 10. TRAIN EACH MODEL
# ==================================================

for model_config in models:

    model_name = model_config["name"]

    model = model_config["model"]

    params = model_config["params"]


    print("\n")
    print("==========================================")
    print(
        "Training:",
        model_name
    )
    print("==========================================")


    # ----------------------------------------------
    # CREATE PIPELINE
    # ----------------------------------------------

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "model",

                model
            )
        ]
    )


    # ----------------------------------------------
    # START MLflow RUN
    # ----------------------------------------------

    with mlflow.start_run(
        run_name=model_name
    ):

        # ------------------------------------------
        # TRAIN
        # ------------------------------------------

        pipeline.fit(
            X_train,
            y_train
        )


        # ------------------------------------------
        # PREDICT
        # ------------------------------------------

        y_pred = pipeline.predict(
            X_test
        )


        # ------------------------------------------
        # METRICS
        # ------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )


        # ------------------------------------------
        # PRINT RESULTS
        # ------------------------------------------

        print(
            "Accuracy:",
            round(accuracy, 4)
        )

        print(
            "Precision:",
            round(precision, 4)
        )

        print(
            "Recall:",
            round(recall, 4)
        )

        print(
            "F1 Score:",
            round(f1, 4)
        )


        # ------------------------------------------
        # LOG PARAMETERS
        # ------------------------------------------

        for parameter_name, parameter_value in params.items():

            mlflow.log_param(
                parameter_name,
                parameter_value
            )


        mlflow.log_param(
            "test_size",
            0.20
        )


        # ------------------------------------------
        # LOG METRICS
        # ------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )


        # ------------------------------------------
        # LOG MODEL
        # ------------------------------------------

        mlflow.sklearn.log_model(

            pipeline,

            name="recommendation_model",

            input_example=X_train.head(1)
        )


        print(
            "MLflow run completed:",
            model_name
        )


print("\n")
print("==========================================")
print("ALL MODELS TRAINED")
print("==========================================")