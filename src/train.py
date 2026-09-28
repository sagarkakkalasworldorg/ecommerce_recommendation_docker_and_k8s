import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/ml_training_data.csv"
)

print("Dataset loaded.")

print(
    "Dataset shape:",
    df.shape
)


# --------------------------------------------------
# DISPLAY TARGET DISTRIBUTION
# --------------------------------------------------

print("\nTarget distribution:")

print(
    df["preferred_category"].value_counts()
)


# --------------------------------------------------
# SEPARATE FEATURES AND TARGET
# --------------------------------------------------

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
    X.head()
)

print("\nTarget:")

print(
    y.head()
)


# --------------------------------------------------
# IDENTIFY CATEGORICAL FEATURES
# --------------------------------------------------

categorical_features = [
    "gender",
    "location"
]


# --------------------------------------------------
# IDENTIFY NUMERICAL FEATURES
# --------------------------------------------------

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


# --------------------------------------------------
# ENCODING
# --------------------------------------------------

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


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

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


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# CREATE ML PIPELINE
# --------------------------------------------------

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


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_pred = pipeline.predict(
    X_test
)


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n--------------------------------")
print("MODEL RESULTS")
print("--------------------------------")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)