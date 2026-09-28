import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)


# ==================================================
# 3. DEFINE CATEGORICAL FEATURES
# ==================================================

categorical_features = [
    "gender",
    "location"
]


# ==================================================
# 4. DEFINE NUMERICAL FEATURES
# ==================================================

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


# ==================================================
# 5. CREATE PREPROCESSOR
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


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ==================================================
# 7. MODEL PARAMETERS
# ==================================================

n_estimators = 100

random_state = 42


# ==================================================
# 8. CREATE RANDOM FOREST
# ==================================================

model = RandomForestClassifier(

    n_estimators=n_estimators,

    random_state=random_state
)


# ==================================================
# 9. CREATE PIPELINE
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
# 10. SET MLflow EXPERIMENT
# ==================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "ecommerce-recommendation"
)


# ==================================================
# 11. START MLflow RUN
# ==================================================

with mlflow.start_run():

# ==================================================
# To set Default- Run Name
# ==================================================    
#
# with mlflow.start_run(
#     run_name="random-forest-100-trees"
# ):    

    print("\n================================")
    print("MLflow Run Started")
    print("================================")

    # ----------------------------------------------
    # TRAIN
    # ----------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "\nModel training completed."
    )


    # ----------------------------------------------
    # PREDICT
    # ----------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )


    # ----------------------------------------------
    # CALCULATE METRICS
    # ----------------------------------------------

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


    # ----------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------

    print("\n================================")
    print("MODEL RESULTS")
    print("================================")

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


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )


    # ==================================================
    # 12. LOG PARAMETERS
    # ==================================================

    print(
        "\nLogging parameters..."
    )

    mlflow.log_param(
        "model_type",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "test_size",
        0.20
    )


    # ==================================================
    # 13. LOG METRICS
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


    # ==================================================
    # 14. LOG MODEL
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
        "\n================================"
    )

    print(
        "MLflow Run Completed"
    )

    print(
        "================================"
    )