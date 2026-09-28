import pandas as pd

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

users_df = pd.read_csv(
    "data/users.csv"
)

products_df = pd.read_csv(
    "data/products.csv"
)

interactions_df = pd.read_csv(
    "data/interactions.csv"
)

preferences_df = pd.read_csv(
    "data/user_preferences.csv"
)

print("Data loaded successfully.")

print(
    "Users:",
    users_df.shape
)

print(
    "Products:",
    products_df.shape
)

print(
    "Interactions:",
    interactions_df.shape
)

# --------------------------------------------------
# JOIN INTERACTIONS WITH PRODUCT INFORMATION
# --------------------------------------------------

interaction_data = interactions_df.merge(
    products_df[
        [
            "product_id",
            "category"
        ]
    ],
    on="product_id",
    how="left"
)

print("\nInteraction data:")
print(
    interaction_data.head()
)

# --------------------------------------------------
# CREATE INTERACTION FEATURES
# --------------------------------------------------

# Convert each action into a numerical count.
#
# Example:
#
# view       -> 1
# click      -> 1
# wishlist   -> 1
# cart       -> 1
# purchase   -> 1

action_features = (
    interaction_data
    .groupby(
        [
            "user_id",
            "category",
            "action"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

print(
    "\nAction features:"
)

print(
    action_features.head()
)

# --------------------------------------------------
# PIVOT DATA
# --------------------------------------------------

feature_data = action_features.pivot_table(
    index="user_id",
    columns=[
        "category",
        "action"
    ],
    values="count",
    fill_value=0
)

# --------------------------------------------------
# FLATTEN COLUMN NAMES
# --------------------------------------------------

feature_data.columns = [
    f"{category}_{action}"
    for category, action
    in feature_data.columns
]

feature_data = feature_data.reset_index()

print(
    "\nFeature data:"
)

print(
    feature_data.head()
)

# --------------------------------------------------
# ADD USER INFORMATION
# --------------------------------------------------

feature_data = users_df.merge(
    feature_data,
    on="user_id",
    how="left"
)

# --------------------------------------------------
# FILL MISSING VALUES
# --------------------------------------------------

feature_data = feature_data.fillna(0)

# --------------------------------------------------
# ADD TARGET
# --------------------------------------------------

feature_data = feature_data.merge(
    preferences_df,
    on="user_id",
    how="left"
)

# --------------------------------------------------
# SAVE TRAINING DATA
# --------------------------------------------------

feature_data.to_csv(
    "data/ml_training_data.csv",
    index=False
)

# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print(
    "\nML training dataset created!"
)

print(
    "Shape:",
    feature_data.shape
)

print(
    "\nColumns:"
)

for column in feature_data.columns:
    print(column)

print(
    "\nSample data:"
)

print(
    feature_data.head()
)