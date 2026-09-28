import pandas as pd
import numpy as np
import random

# Reproducibility
np.random.seed(42)
random.seed(42)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

NUM_USERS = 1000
NUM_PRODUCTS = 500
NUM_INTERACTIONS = 20000

# --------------------------------------------------
# PRODUCT CATEGORIES
# --------------------------------------------------

categories = {
    "Electronics": [
        "Smartphone",
        "Laptop",
        "Tablet",
        "Smartwatch",
        "Headphones"
    ],
    "Gaming": [
        "Gaming Laptop",
        "Gaming Mouse",
        "Gaming Keyboard",
        "Gaming Headset",
        "Gaming Monitor"
    ],
    "Fashion": [
        "T-Shirt",
        "Jeans",
        "Dress",
        "Shoes",
        "Jacket",
        "Handbag"
    ],
    "Automotive": [
        "Car Dash Camera",
        "Car Floor Mat",
        "Car Charger",
        "Car Vacuum",
        "Car Phone Holder"
    ],
    "Beauty": [
        "Makeup Kit",
        "Face Cream",
        "Perfume",
        "Skincare Set",
        "Hair Dryer"
    ],
    "Sports": [
        "Running Shoes",
        "Football",
        "Yoga Mat",
        "Dumbbells",
        "Cricket Bat"
    ],
    "Home": [
        "Coffee Machine",
        "Table Lamp",
        "Bedsheet",
        "Vacuum Cleaner",
        "Air Purifier"
    ],
    "Books": [
        "Python Book",
        "AI Book",
        "Cloud Computing Book",
        "Business Book",
        "Self Help Book"
    ]
}

# --------------------------------------------------
# GENERATE USERS
# --------------------------------------------------

users = []

genders = ["Male", "Female"]
locations = ["India", "Portugal", "UK", "Germany", "USA"]

for i in range(1, NUM_USERS + 1):

    user = {
        "user_id": f"U{i:04d}",
        "age": random.randint(18, 60),
        "gender": random.choice(genders),
        "location": random.choice(locations)
    }

    users.append(user)

users_df = pd.DataFrame(users)

# --------------------------------------------------
# GENERATE PRODUCTS
# --------------------------------------------------

products = []

product_id = 1

for category, product_names in categories.items():

    for product_name in product_names:

        # Generate multiple products for each product type
        for i in range(10):

            product = {
                "product_id": f"P{product_id:04d}",
                "product_name": product_name,
                "category": category,
                "price": round(random.uniform(20, 1500), 2),
                "rating": round(random.uniform(3.0, 5.0), 1)
            }

            products.append(product)

            product_id += 1

products_df = pd.DataFrame(products)

# Keep only requested number of products
products_df = products_df.head(NUM_PRODUCTS)

# --------------------------------------------------
# GENERATE USER-PRODUCT INTERACTIONS
# --------------------------------------------------

actions = [
    "view",
    "click",
    "wishlist",
    "cart",
    "purchase"
]

interactions = []

for i in range(NUM_INTERACTIONS):

    user = random.choice(users_df["user_id"].tolist())

    product = random.choice(products_df["product_id"].tolist())

    action = random.choices(
        actions,
        weights=[50, 25, 10, 10, 5]
    )[0]

    interaction = {
        "interaction_id": f"I{i + 1:06d}",
        "user_id": user,
        "product_id": product,
        "action": action
    }

    interactions.append(interaction)

interactions_df = pd.DataFrame(interactions)

# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

users_df.to_csv("data/users.csv", index=False)

products_df.to_csv("data/products.csv", index=False)

interactions_df.to_csv(
    "data/interactions.csv",
    index=False
)

# --------------------------------------------------
# DISPLAY INFORMATION
# --------------------------------------------------

print("Data generation completed!")

print("\nUsers:")
print(users_df.head())

print("\nProducts:")
print(products_df.head())

print("\nInteractions:")
print(interactions_df.head())

print("\nDataset sizes:")

print("Users:", len(users_df))

print("Products:", len(products_df))

print("Interactions:", len(interactions_df))