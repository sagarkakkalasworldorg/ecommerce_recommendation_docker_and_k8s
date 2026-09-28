import streamlit as st
import requests


# ==================================================
# CONFIGURATION
# ==================================================

BENTOML_URL = "http://localhost:3000/predict"


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="E-commerce Recommendation System",
    page_icon="🛒",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("🛒 E-commerce Recommendation System")

st.write(
    "Enter the customer's behaviour and get "
    "a recommended category and products."
)


# ==================================================
# USER INFORMATION
# ==================================================

st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:

    gender = st.selectbox(
        "Gender",
        [
            "male",
            "female"
        ]
    )

with col3:

    location = st.text_input(
        "Location",
        value="porto"
    )


# ==================================================
# CATEGORY BEHAVIOUR
# ==================================================

st.header("Customer Behaviour")


categories = [
    "Automotive",
    "Beauty",
    "Books",
    "Electronics",
    "Fashion",
    "Gaming",
    "Home",
    "Sports"
]


user_data = {

    "age": age,

    "gender": gender,

    "location": location
}


# ==================================================
# CATEGORY INPUTS
# ==================================================

for category in categories:

    st.subheader(category)

    col1, col2, col3 = st.columns(3)

    with col1:

        carts = st.number_input(
            f"{category} Carts",
            min_value=0,
            value=0,
            key=f"{category}_carts"
        )

    with col2:

        clicks = st.number_input(
            f"{category} Clicks",
            min_value=0,
            value=0,
            key=f"{category}_clicks"
        )

    with col3:

        purchases = st.number_input(
            f"{category} Purchases",
            min_value=0,
            value=0,
            key=f"{category}_purchases"
        )


    col4, col5, col6 = st.columns(3)

    with col4:

        views = st.number_input(
            f"{category} Views",
            min_value=0,
            value=0,
            key=f"{category}_views"
        )

    with col5:

        wishlists = st.number_input(
            f"{category} Wishlists",
            min_value=0,
            value=0,
            key=f"{category}_wishlists"
        )

    with col6:

        engagement_score = st.number_input(
            f"{category} Engagement Score",
            min_value=0,
            value=0,
            key=f"{category}_engagement_score"
        )


    user_data[
        f"{category}_carts"
    ] = carts

    user_data[
        f"{category}_clicks"
    ] = clicks

    user_data[
        f"{category}_purchases"
    ] = purchases

    user_data[
        f"{category}_views"
    ] = views

    user_data[
        f"{category}_wishlists"
    ] = wishlists

    user_data[
        f"{category}_engagement_score"
    ] = engagement_score


# ==================================================
# PREDICTION BUTTON
# ==================================================

st.divider()

if st.button(
    "🔮 Get Recommendation",
    type="primary"
):

    # ----------------------------------------------
    # Create request payload
    # ----------------------------------------------

    payload = {

        "user_data": user_data

    }


    # ----------------------------------------------
    # Call BentoML
    # ----------------------------------------------

    try:

        response = requests.post(

            BENTOML_URL,

            json=payload,

            timeout=30

        )


        # ------------------------------------------
        # Check response
        # ------------------------------------------

        if response.status_code == 200:

            result = response.json()


            # --------------------------------------
            # Display predicted category
            # --------------------------------------

            st.success(
                "Recommendation generated successfully!"
            )


            st.header(
                "🎯 Recommended Category"
            )

            st.subheader(
                result[
                    "recommended_category"
                ]
            )


            # --------------------------------------
            # Display products
            # --------------------------------------

            st.header(
                "🛍️ Recommended Products"
            )


            products = result.get(
                "products",
                []
            )


            if products:

                columns = st.columns(
                    len(products)
                )


                for column, product in zip(
                    columns,
                    products
                ):

                    with column:

                        st.markdown(
                            f"### {product['product_name']}"
                        )

                        st.write(
                            f"**Product ID:** "
                            f"{product['product_id']}"
                        )

                        st.write(
                            f"**Category:** "
                            f"{product['category']}"
                        )

                        st.write(
                            f"**Price:** "
                            f"€{product['price']}"
                        )

                        st.write(
                            f"**Rating:** "
                            f"⭐ {product['rating']}"
                        )


            else:

                st.warning(
                    "No products were returned."
                )


        else:

            st.error(
                f"BentoML returned "
                f"status code {response.status_code}"
            )

            st.json(
                response.json()
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to BentoML. "
            "Make sure BentoML is running on "
            "http://localhost:3000"
        )


    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )