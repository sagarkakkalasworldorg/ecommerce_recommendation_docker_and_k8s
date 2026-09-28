import pandas as pd
import random
import numpy as np


# ==================================================
# CONFIGURATION
# ==================================================

random.seed(42)
np.random.seed(42)


# ==================================================
# FILE PATHS
# ==================================================

USERS_FILE = "data/users.csv"

PRODUCTS_FILE = "data/products.csv"

INTERACTIONS_FILE = "data/realistic_interactions.csv"

PREFERENCES_FILE = "data/realistic_user_preferences.csv"


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
# LOAD PRODUCTS
# ==================================================

print("\nLoading products...")

products = pd.read_csv(
    PRODUCTS_FILE
)

print(
    "Products:",
    len(products)
)


# ==================================================
# DISPLAY PRODUCT COLUMNS
# ==================================================

print("\nProduct columns:")

print(
    products.columns.tolist()
)


# ==================================================
# FIND PRODUCT ID COLUMN
# ==================================================

possible_product_id_columns = [

    "ProductID",

    "product_id",

    "productId",

    "id"

]


product_id_column = None


for column in possible_product_id_columns:

    if column in products.columns:

        product_id_column = column

        break


if product_id_column is None:

    raise ValueError(
        "\nCould not find Product ID column.\n"
        "Available columns: "
        + str(products.columns.tolist())
    )


# ==================================================
# FIND CATEGORY COLUMN
# ==================================================

possible_category_columns = [

    "Category",

    "category",

    "ProductCategory",

    "product_category",

    "product_category_name"

]


category_column = None


for column in possible_category_columns:

    if column in products.columns:

        category_column = column

        break


if category_column is None:

    raise ValueError(
        "\nCould not find product category column.\n"
        "Available columns: "
        + str(products.columns.tolist())
    )


print(
    "\nUsing Product ID column:",
    product_id_column
)

print(
    "Using Category column:",
    category_column
)


# ==================================================
# PRODUCT CATEGORIES
# ==================================================

categories = products[
    category_column
].dropna().unique().tolist()


print("\nCategories:")

for category in categories:

    print(
        "-",
        category
    )


# ==================================================
# CATEGORY VALIDATION
# ==================================================

if len(categories) < 3:

    raise ValueError(
        "At least 3 product categories are required."
    )


# ==================================================
# GENERATE USER PREFERENCES
# ==================================================

print(
    "\nGenerating user preferences..."
)


user_preferences = []


for _, user in users.iterrows():

    user_id = user["user_id"]


    # ----------------------------------------------
    # PRIMARY INTEREST
    # ----------------------------------------------

    primary_category = random.choice(
        categories
    )


    # ----------------------------------------------
    # SECONDARY INTERESTS
    # ----------------------------------------------

    remaining_categories = [

        category

        for category in categories

        if category != primary_category

    ]


    secondary_categories = random.sample(

        remaining_categories,

        2

    )


    user_preferences.append({

        "user_id":
            user_id,

        "primary_category":
            primary_category,

        "secondary_category_1":
            secondary_categories[0],

        "secondary_category_2":
            secondary_categories[1]

    })


preferences_df = pd.DataFrame(
    user_preferences
)


# ==================================================
# SAVE USER PREFERENCES
# ==================================================

preferences_df.to_csv(

    PREFERENCES_FILE,

    index=False

)


print(
    "\nUser preferences saved to:"
)

print(
    PREFERENCES_FILE
)


# ==================================================
# GENERATE INTERACTIONS
# ==================================================

print(
    "\nGenerating realistic interactions..."
)


interactions = []


# ==================================================
# USER BEHAVIOR MODEL
# ==================================================

# Primary category:
#
# 45%
#
# Secondary category 1:
#
# 20%
#
# Secondary category 2:
#
# 20%
#
# Random category:
#
# 15%
#
# This creates imperfect behavior.


for _, preference in preferences_df.iterrows():

    user_id = preference[
        "user_id"
    ]

    primary_category = preference[
        "primary_category"
    ]

    secondary_category_1 = preference[
        "secondary_category_1"
    ]

    secondary_category_2 = preference[
        "secondary_category_2"
    ]


    # ==================================================
    # GENERATE 40 INTERACTIONS
    # ==================================================

    for _ in range(40):

        random_value = random.random()


        # ----------------------------------------------
        # SELECT CATEGORY
        # ----------------------------------------------

        if random_value < 0.45:

            category = primary_category


        elif random_value < 0.65:

            category = secondary_category_1


        elif random_value < 0.85:

            category = secondary_category_2


        else:

            other_categories = [

                category_name

                for category_name in categories

                if category_name not in [

                    primary_category,

                    secondary_category_1,

                    secondary_category_2

                ]

            ]


            category = random.choice(
                other_categories
            )


        # ==================================================
        # SELECT PRODUCT
        # ==================================================

        category_products = products[

            products[category_column]
            == category

        ]


        product = category_products.sample(
            1
        ).iloc[0]


        product_id = product[
            product_id_column
        ]


        # ==================================================
        # SELECT INTERACTION
        # ==================================================

        interaction_type = random.choices(

            [

                "view",

                "click",

                "wishlist",

                "cart",

                "purchase"

            ],

            weights=[

                55,

                25,

                10,

                7,

                3

            ],

            k=1

        )[0]


        # ==================================================
        # INTERACTION WEIGHT
        # ==================================================

        interaction_weight = {

            "view": 1,

            "click": 2,

            "wishlist": 3,

            "cart": 4,

            "purchase": 5

        }[
            interaction_type
        ]


        # ==================================================
        # SAVE INTERACTION
        # ==================================================

        interactions.append({

            "user_id":
                user_id,

            "product_id":
                product_id,

            "category":
                category,

            "interaction":
                interaction_type,

            "interaction_weight":
                interaction_weight

        })


# ==================================================
# CREATE DATAFRAME
# ==================================================

interactions_df = pd.DataFrame(
    interactions
)


# ==================================================
# SAVE INTERACTIONS
# ==================================================

interactions_df.to_csv(

    INTERACTIONS_FILE,

    index=False

)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print(
    "\nInteractions generated:",
    len(interactions_df)
)


print(
    "\nInteraction distribution:"
)

print(
    interactions_df[
        "interaction"
    ].value_counts()
)


print(
    "\nCategory distribution:"
)

print(
    interactions_df[
        "category"
    ].value_counts()
)


print(
    "\nSample interactions:"
)

print(
    interactions_df.head(10)
)


# ==================================================
# COMPLETE
# ==================================================

print(
    "\n=========================================="
)

print(
    "REALISTIC USER BEHAVIOR GENERATED"
)

print(
    "=========================================="
)

print(
    "\nFiles created:"
)

print(
    INTERACTIONS_FILE
)

print(
    PREFERENCES_FILE
)

print(
    "\n=========================================="
)