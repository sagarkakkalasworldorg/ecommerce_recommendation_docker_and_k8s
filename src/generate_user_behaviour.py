import pandas as pd
import random

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

NUM_INTERACTIONS = 20000

random.seed(42)

# --------------------------------------------------
# LOAD EXISTING DATA
# --------------------------------------------------

users_df = pd.read_csv(
    "data/users.csv"
)

products_df = pd.read_csv(
    "data/products.csv"
)

# --------------------------------------------------
# AVAILABLE CATEGORIES
# --------------------------------------------------

categories = products_df[
    "category"
].unique().tolist()

print("Available categories:")
print(categories)

# --------------------------------------------------
# GENERATE HIDDEN USER PREFERENCES
# --------------------------------------------------

user_preferences = {}

for user_id in users_df["user_id"]:

    preferred_category = random.choice(
        categories
    )

    user_preferences[user_id] = preferred_category


# --------------------------------------------------
# SAVE USER PREFERENCES
# --------------------------------------------------

preferences_df = pd.DataFrame(
    [
        {
            "user_id": user_id,
            "preferred_category": category
        }
        for user_id, category
        in user_preferences.items()
    ]
)

preferences_df.to_csv(
    "data/user_preferences.csv",
    index=False
)

print(
    "\nUser preferences generated successfully."
)

print(
    preferences_df.head()
)

# --------------------------------------------------
# GENERATE REALISTIC INTERACTIONS
# --------------------------------------------------

actions = [
    "view",
    "click",
    "wishlist",
    "cart",
    "purchase"
]

action_weights = [
    50,
    25,
    10,
    10,
    5
]

interactions = []

user_ids = users_df[
    "user_id"
].tolist()

for i in range(NUM_INTERACTIONS):

    # ----------------------------------------------
    # Select a random user
    # ----------------------------------------------

    user_id = random.choice(
        user_ids
    )

    # ----------------------------------------------
    # Get user's hidden preference
    # ----------------------------------------------

    preferred_category = user_preferences[
        user_id
    ]

    # ----------------------------------------------
    # 80% of the time:
    # choose preferred category
    #
    # 20% of the time:
    # choose random category
    # ----------------------------------------------

    if random.random() < 0.80:

        category_products = products_df[
            products_df["category"]
            == preferred_category
        ]

        product = category_products.sample(
            n=1
        ).iloc[0]

    else:

        product = products_df.sample(
            n=1
        ).iloc[0]

    # ----------------------------------------------
    # Select interaction type
    # ----------------------------------------------

    action = random.choices(
        actions,
        weights=action_weights,
        k=1
    )[0]

    # ----------------------------------------------
    # Create interaction record
    # ----------------------------------------------

    interaction = {
        "interaction_id": f"I{i + 1:06d}",
        "user_id": user_id,
        "product_id": product["product_id"],
        "action": action
    }

    interactions.append(
        interaction
    )


# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

interactions_df = pd.DataFrame(
    interactions
)

# --------------------------------------------------
# SAVE INTERACTIONS
# --------------------------------------------------

interactions_df.to_csv(
    "data/interactions.csv",
    index=False
)

print(
    "\nRealistic interactions generated successfully."
)

print(
    interactions_df.head()
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nDataset summary:")

print(
    "Users:",
    len(users_df)
)

print(
    "Products:",
    len(products_df)
)

print(
    "Interactions:",
    len(interactions_df)
)

print(
    "\nFiles generated:"
)

print(
    "data/user_preferences.csv"
)

print(
    "data/interactions.csv"
)