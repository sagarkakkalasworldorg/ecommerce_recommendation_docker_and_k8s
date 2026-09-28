import pandas as pd
import time
import mlflow
import mlflow.sklearn
import os


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
    f1_score,
    classification_report
)


# ==================================================
# 1. LOAD REALISTIC DATASET
# ==================================================

print("\nLoading realistic dataset...")

df = pd.read_csv(
    "data/realistic_ml_training_data.csv"
)

print(
    "Dataset shape:",
    df.shape
)


# ==================================================
# 2. DISPLAY DATA
# ==================================================

print("\nDataset columns:")

print(
    df.columns.tolist()
)


print("\nTarget distribution:")

print(
    df["preferred_category"].value_counts()
)


# ==================================================
# 3. SEPARATE FEATURES AND TARGET
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


print("\nFeatures:")

print(
    X.columns.tolist()
)


print("\nTarget:")

print(
    y.name
)


# ==================================================
# 4. CATEGORICAL FEATURES
# ==================================================

categorical_features = [

    "gender",

    "location"

]


# ==================================================
# 5. NUMERICAL FEATURES
# ==================================================

numerical_features = [

    column

    for column in X.columns

    if column not in categorical_features

]


print("\nCategorical features:")

print(
    categorical_features
)


print("\nNumber of numerical features:")

print(
    len(numerical_features)
)


# ==================================================
# 6. PREPROCESSOR
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
# 7. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining samples:")

print(
    len(X_train)
)


print("\nTesting samples:")

print(
    len(X_test)
)


# ==================================================
# 8. MLflow CONFIGURATION
# ==================================================


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    "ecommerce-recommendation-realistic"
)


# ==================================================
# 9. MODELS
# ==================================================

models = [

    {
        "name": "logistic-regression",

        "model": LogisticRegression(
            max_iter=1000
        ),

        "params": {

            "model_type":
                "LogisticRegression",

            "max_iter":
                1000

        }

    },


    {
        "name": "decision-tree",

        "model": DecisionTreeClassifier(

            max_depth=10,

            random_state=42

        ),

        "params": {

            "model_type":
                "DecisionTreeClassifier",

            "max_depth":
                10,

            "random_state":
                42

        }

    },


    {
        "name": "random-forest-50",

        "model": RandomForestClassifier(

            n_estimators=50,

            random_state=42

        ),

        "params": {

            "model_type":
                "RandomForestClassifier",

            "n_estimators":
                50,

            "random_state":
                42

        }

    },


    {
        "name": "random-forest-100",

        "model": RandomForestClassifier(

            n_estimators=100,

            random_state=42

        ),

        "params": {

            "model_type":
                "RandomForestClassifier",

            "n_estimators":
                100,

            "random_state":
                42

        }

    },


    {
        "name": "random-forest-200",

        "model": RandomForestClassifier(

            n_estimators=200,

            random_state=42

        ),

        "params": {

            "model_type":
                "RandomForestClassifier",

            "n_estimators":
                200,

            "random_state":
                42

        }

    }

]


# ==================================================
# 10. TRAIN MODELS
# ==================================================

for model_config in models:

    model_name = model_config[
        "name"
    ]

    model = model_config[
        "model"
    ]

    params = model_config[
        "params"
    ]


    print("\n")
    print(
        "=========================================="
    )

    print(
        "Training:",
        model_name
    )

    print(
        "=========================================="
    )


    # ==================================================
    # CREATE PIPELINE
    # ==================================================

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


    # ==================================================
    # START MLflow RUN
    # ==================================================

    with mlflow.start_run(
        run_name=model_name
    ):


        # ==================================================
        # TRAIN
        # ==================================================

        training_start_time = time.time()

        pipeline.fit(

            X_train,

            y_train

        )

        training_time = (
            time.time()
            - training_start_time
        )


        print(
            "Model training completed."
        )

        print(
            "Training time:",
            round(training_time, 4),
            "seconds"
        )

        # ==================================================
        # PREDICT
        # ==================================================


        prediction_start_time = time.time()

        y_pred = pipeline.predict(
            X_test
        )

        prediction_time = (
            time.time()
            - prediction_start_time
        )


        print(
            "Prediction completed."
        )

        print(
            "Prediction time:",
            round(prediction_time, 4),
            "seconds"
        )

        # ==================================================
        # METRICS
        # ==================================================

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


        # ==================================================
        # PRINT RESULTS
        # ==================================================

        print("\nMODEL RESULTS")

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


        print(
            "\nClassification Report:"
        )

        print(

            classification_report(

                y_test,

                y_pred

            )

        )


        # ==================================================
        # LOG PARAMETERS
        # ==================================================

        print(
            "\nLogging parameters..."
        )


        for parameter_name, parameter_value in params.items():

            mlflow.log_param(

                parameter_name,

                parameter_value

            )


        mlflow.log_param(

            "test_size",

            0.20

        )


        mlflow.log_param(

            "random_state_split",

            42

        )


        # ==================================================
        # LOG METRICS
        # ==================================================

        print(
            "Logging metrics..."
        )


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

        mlflow.log_metric(
            "training_time_seconds",
            training_time
        )

        mlflow.log_metric(
            "prediction_time_seconds",
            prediction_time
        )


        # ==================================================
        # LOG MODEL
        # ==================================================

        print(
            "Logging model..."
        )


        mlflow.sklearn.log_model(

            pipeline,

            name="recommendation_model",

            input_example=X_train.head(1)

        )


        print(
            "\nMLflow run completed:",
            model_name
        )


# ==================================================
# COMPLETE
# ==================================================

print("\n")

print(
    "=========================================="
)

print(
    "ALL REALISTIC MODELS TRAINED"
)

print(
    "=========================================="
)