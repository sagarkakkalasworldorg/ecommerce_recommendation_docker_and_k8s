import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

PRODUCTS_FILE = "data/products.csv"


# ==================================================
# LOAD PRODUCT CATALOG
# ==================================================

print("\nLoading product catalog...")

products = pd.read_csv(
    PRODUCTS_FILE
)

print(
    "Products loaded:",
    len(products)
)


# ==================================================
# DISPLAY AVAILABLE CATEGORIES
# ==================================================

categories = products[
    "category"
].unique().tolist()


print("\nAvailable categories:")

for category in categories:

    print(
        "-",
        category
    )


# ==================================================
# RECOMMEND PRODUCTS
# ==================================================

def recommend_products(
    category,
    number_of_products=5
):

    # ----------------------------------------------
    # Filter products by category
    # ----------------------------------------------

    category_products = products[
        products["category"] == category
    ]


    # ----------------------------------------------
    # Check if category exists
    # ----------------------------------------------

    if category_products.empty:

        return []


    # ----------------------------------------------
    # Select products
    # ----------------------------------------------

    recommendations = category_products.sample(
        min(
            number_of_products,
            len(category_products)
        )
    )


    # ----------------------------------------------
    # Return selected products
    # ----------------------------------------------

    return recommendations.to_dict(
        orient="records"
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    # ----------------------------------------------
    # Simulate ML prediction
    # ----------------------------------------------

    predicted_category = "Electronics"


    # ----------------------------------------------
    # Get recommendations
    # ----------------------------------------------

    recommended_products = recommend_products(

        predicted_category,

        number_of_products=5
    )


    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    print(
        "\n================================"
    )

    print(
        "Recommended Category:",
        predicted_category
    )

    print(
        "================================"
    )


    print(
        "\nRecommended Products:\n"
    )


    for product in recommended_products:

        print(
            f"Product ID: {product['product_id']}"
        )

        print(
            f"Product Name: {product['product_name']}"
        )

        print(
            f"Category: {product['category']}"
        )

        print(
            f"Price: €{product['price']}"
        )

        print(
            f"Rating: {product['rating']}"
        )

        print(
            "--------------------------------"
        )