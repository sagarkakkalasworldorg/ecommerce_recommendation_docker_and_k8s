import pandas as pd


# ==================================================
# FILE PATHS
# ==================================================

USERS_FILE = "data/users.csv"

INTERACTIONS_FILE = (
    "data/realistic_interactions.csv"
)

PREFERENCES_FILE = (
    "data/realistic_user_preferences.csv"
)

OUTPUT_FILE = (
    "data/realistic_ml_training_data.csv"
)


# ==================================================
# LOAD USERS
# ==================================================

print("\nLoading users...")

users = pd.read_csv(
    USERS_FILE
)

print(
    "Users:",
    len(users)
)


# ==================================================
# LOAD INTERACTIONS
# ==================================================

print("\nLoading realistic interactions...")

interactions = pd.read_csv(
    INTERACTIONS_FILE
)

print(
    "Interactions:",
    len(interactions)
)


# ==================================================
# LOAD USER PREFERENCES
# ==================================================

print("\nLoading user preferences...")

preferences = pd.read_csv(
    PREFERENCES_FILE
)

print(
    "User preferences:",
    len(preferences)
)


# ==================================================
# DISPLAY COLUMNS
# ==================================================

print("\nUser columns:")

print(
    users.columns.tolist()
)


print("\nInteraction columns:")

print(
    interactions.columns.tolist()
)


print("\nPreference columns:")

print(
    preferences.columns.tolist()
)


# ==================================================
# MERGE USER INFORMATION
# ==================================================

print(
    "\nMerging user information..."
)


user_data = users.merge(

    preferences,

    on="user_id",

    how="left"

)


# ==================================================
# CREATE USER BEHAVIOR FEATURES
# ==================================================

print(
    "\nCreating behavior features..."
)


# --------------------------------------------------
# Create interaction counts
# --------------------------------------------------

behavior_counts = (

    interactions

    .groupby(
        [
            "user_id",
            "category",
            "interaction"
        ]
    )

    .size()

    .reset_index(
        name="count"
    )

)


# ==================================================
# PIVOT INTERACTION DATA
# ==================================================

behavior_pivot = (

    behavior_counts

    .pivot_table(

        index=[
            "user_id",
            "category"
        ],

        columns="interaction",

        values="count",

        fill_value=0

    )

    .reset_index()

)


# ==================================================
# MAKE COLUMN NAMES CLEAN
# ==================================================

behavior_pivot.columns.name = None


# ==================================================
# ENSURE ALL INTERACTION TYPES EXIST
# ==================================================

interaction_types = [

    "view",

    "click",

    "wishlist",

    "cart",

    "purchase"

]


for interaction_type in interaction_types:

    if interaction_type not in behavior_pivot.columns:

        behavior_pivot[
            interaction_type
        ] = 0


# ==================================================
# RENAME INTERACTION COLUMNS
# ==================================================

behavior_pivot = behavior_pivot.rename(

    columns={

        "view":
            "views",

        "click":
            "clicks",

        "wishlist":
            "wishlists",

        "cart":
            "carts",

        "purchase":
            "purchases"

    }

)


# ==================================================
# CALCULATE CATEGORY SCORE
# ==================================================

print(
    "\nCalculating category engagement scores..."
)


behavior_pivot["engagement_score"] = (

    behavior_pivot["views"] * 1

    + behavior_pivot["clicks"] * 2

    + behavior_pivot["wishlists"] * 3

    + behavior_pivot["carts"] * 4

    + behavior_pivot["purchases"] * 5

)


# ==================================================
# PIVOT CATEGORIES INTO FEATURES
# ==================================================

print(
    "\nCreating category-level features..."
)


feature_columns = [

    "views",

    "clicks",

    "wishlists",

    "carts",

    "purchases",

    "engagement_score"

]


category_features = (

    behavior_pivot

    .pivot(

        index="user_id",

        columns="category",

        values=feature_columns

    )

)


# ==================================================
# FLATTEN MULTI-INDEX COLUMNS
# ==================================================

category_features.columns = [

    f"{category}_{feature}"

    for feature, category
    in category_features.columns

]


category_features = (

    category_features

    .reset_index()

)


# ==================================================
# FILL MISSING VALUES
# ==================================================

category_features = (

    category_features

    .fillna(0)

)


# ==================================================
# MERGE USER DATA
# ==================================================

print(
    "\nCombining user and behavior data..."
)


training_data = user_data.merge(

    category_features,

    on="user_id",

    how="left"

)


# ==================================================
# CREATE TARGET
# ==================================================

# IMPORTANT:
#
# We use the PRIMARY CATEGORY as the target.
#
# The ML model will NOT receive this column
# as an input feature.
#
# Later, our training script will remove it
# from X and use it as y.


training_data["preferred_category"] = (

    training_data[
        "primary_category"
    ]

)


# ==================================================
# REMOVE ORIGINAL PREFERENCE COLUMNS
# ==================================================

# We don't want the model to directly see:
#
# primary_category
# secondary_category_1
# secondary_category_2
#
# Otherwise the model would simply receive
# the answer.
#
# This is an important example of preventing
# target leakage.


training_data = training_data.drop(

    columns=[

        "primary_category",

        "secondary_category_1",

        "secondary_category_2"

    ]

)


# ==================================================
# SAVE DATASET
# ==================================================

training_data.to_csv(

    OUTPUT_FILE,

    index=False

)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print(
    "\n=========================================="
)

print(
    "REALISTIC ML DATASET CREATED"
)

print(
    "=========================================="
)


print(
    "\nDataset shape:"
)

print(
    training_data.shape
)


print(
    "\nColumns:"
)

print(
    training_data.columns.tolist()
)


print(
    "\nTarget distribution:"
)

print(
    training_data[
        "preferred_category"
    ].value_counts()
)


print(
    "\nSample data:"
)

print(
    training_data.head()
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_FILE
)


print(
    "\n=========================================="
)