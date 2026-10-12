# BigMart Sales Prediction Dashboard
# This app loads the saved machine-learning model.
# Users enter product and outlet details to predict sales.





# Import libraries for the dashboard and model
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Find the project folder and saved model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "bigmart_tuned_rf.joblib"

# Configure the browser tab and layout
st.set_page_config(
    page_title="BigMart Sales Predictor",
    page_icon="🛒",
    layout="wide"
)

# Load the trained model only once
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Create an empty prediction history for this session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Add sidebar information
with st.sidebar:
    st.title("🛒 BigMart")
    st.write("Sales Prediction System")
    st.divider()
    st.subheader("About this project")
    st.write(
        "This machine-learning project estimates "
        "sales for a product at a particular outlet."
    )
    st.caption("Model: Tuned Random Forest")
    st.caption("Validation R²: 0.6123")

# Display the main heading
st.title("BigMart Sales Prediction")
st.write(
    "Enter product and outlet details to estimate item sales."
)

# Store the known details for each BigMart outlet
outlet_details = {
    "OUT010": {
        "size": "Small",
        "location": "Tier 3",
        "type": "Grocery Store",
        "year": 1998
    },
    "OUT013": {
        "size": "High",
        "location": "Tier 3",
        "type": "Supermarket Type1",
        "year": 1987
    },
    "OUT017": {
        "size": "Small",
        "location": "Tier 2",
        "type": "Supermarket Type1",
        "year": 2007
    },
    "OUT018": {
        "size": "Medium",
        "location": "Tier 3",
        "type": "Supermarket Type2",
        "year": 2009
    },
    "OUT019": {
        "size": "Small",
        "location": "Tier 1",
        "type": "Grocery Store",
        "year": 1985
    },
    "OUT027": {
        "size": "Medium",
        "location": "Tier 3",
        "type": "Supermarket Type3",
        "year": 1985
    },
    "OUT035": {
        "size": "Small",
        "location": "Tier 2",
        "type": "Supermarket Type1",
        "year": 2004
    },
    "OUT045": {
        "size": "Small",
        "location": "Tier 2",
        "type": "Supermarket Type1",
        "year": 2002
    },
    "OUT046": {
        "size": "Small",
        "location": "Tier 1",
        "type": "Supermarket Type1",
        "year": 1997
    },
    "OUT049": {
        "size": "Medium",
        "location": "Tier 1",
        "type": "Supermarket Type1",
        "year": 1999
    }
}

# Organize the form into two columns
with st.form("sales_form"):
    st.subheader("📦 Product Information")

    col1, col2 = st.columns(2)

    with col1:
        item_id = st.text_input("Item Identifier", "FD001")
        item_weight = st.number_input(
            "Item Weight", min_value=0.0, value=12.6
        )
        fat_content = st.selectbox(
            "Fat Content", ["Low Fat", "Regular"]
        )
        item_type = st.selectbox(
            "Item Type",
            [
                "Dairy", "Soft Drinks", "Meat",
                "Fruits and Vegetables", "Household",
                "Baking Goods", "Snack Foods", "Frozen Foods",
                "Breakfast", "Health and Hygiene", "Hard Drinks",
                "Canned", "Breads", "Starchy Foods",
                "Others", "Seafood"
            ]
        )

    with col2:
        item_visibility = st.number_input(
            "Item Visibility",
            min_value=0.0, max_value=1.0, value=0.05
        )
        item_mrp = st.number_input(
            "Item MRP", min_value=0.0, value=150.0
        )

    st.divider()
    st.subheader("🏬 Outlet Information")

    col3, col4 = st.columns(2)

    with col3:
        outlet_id = st.selectbox(
            "Outlet Identifier",
            [
                "OUT010", "OUT013", "OUT017", "OUT018",
                "OUT019", "OUT027", "OUT035", "OUT045",
                "OUT046", "OUT049"
            ]
        )
        establishment_year = st.selectbox(
            "Establishment Year",
            [1985, 1987, 1997, 1998, 1999, 2002, 2004, 2007, 2009]
        )
        outlet_size = st.selectbox(
            "Outlet Size", ["Small", "Medium", "High"]
        )

    with col4:
        location_type = st.selectbox(
            "Outlet Location",
            ["Tier 1", "Tier 2", "Tier 3"]
        )
        outlet_type = st.selectbox(
            "Outlet Type",
            [
                "Grocery Store", "Supermarket Type1",
                "Supermarket Type2", "Supermarket Type3"
            ]
        )

    # Display a full-width prediction button
    submitted = st.form_submit_button(
        "🔮 Predict Sales", use_container_width=True
    )

# Store the average sales calculated during model development
AVERAGE_SALES = 2181.288914

# Predict when the user submits the form
if submitted:
    # Arrange the user inputs into a DataFrame
    input_data = pd.DataFrame([{
        "Item_Identifier": item_id.strip().upper(),
        "Item_Weight": item_weight,
        "Item_Fat_Content": fat_content,
        "Item_Visibility": item_visibility,
        "Item_Type": item_type,
        "Item_MRP": item_mrp,
        "Outlet_Identifier": outlet_id,
        "Outlet_Establishment_Year": establishment_year,
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": location_type,
        "Outlet_Type": outlet_type
    }])

    # Predict sales using the trained model
    prediction = model.predict(input_data)[0]

    # Save the current prediction in the session history
    st.session_state.prediction_history.append({
        "Item Identifier": item_id.strip().upper(),
        "Outlet Identifier": outlet_id,
        "Predicted Sales": round(float(prediction), 2)
    })

    # Get the average sales from the training dataset
    # Use the saved average instead of reading Train.csv
    average_sales = AVERAGE_SALES

    # Display the prediction result
    st.divider()
    st.subheader("📊 Your Prediction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Item Outlet Sales",
            f"{prediction:,.2f}"
        )

    with result_col2:
        st.metric(
            "Training Data Average",
            f"{average_sales:,.2f}"
        )

    # Display a chart comparing both values
    chart_data = pd.DataFrame({
        "Sales Type": ["Training Average", "Your Prediction"],
        "Sales Value": [average_sales, prediction]
    })

    st.subheader("📈 Sales Comparison")
    st.bar_chart(
        chart_data.set_index("Sales Type"),
        y="Sales Value"
    )

    # Explain the result
    difference = prediction - average_sales

    if difference > 0:
        st.info(
            f"The prediction is {difference:,.2f} "
            "above the overall training-data average."
        )
    elif difference < 0:
        st.info(
            f"The prediction is {abs(difference):,.2f} "
            "below the overall training-data average."
        )
    else:
        st.info("The prediction equals the training-data average.")

    st.caption(
        "The comparison uses the overall training-data average. "
        "It does not guarantee actual future sales."
    )

# Display prediction history
st.divider()
st.subheader("📋 Prediction History")

# Show previous predictions if available
if st.session_state.prediction_history:
    history_data = pd.DataFrame(
        st.session_state.prediction_history
    )

    # Display the history table
    st.dataframe(history_data, use_container_width=True)

    # Compare predictions in a chart
    st.subheader("📈 Prediction History Chart")
    st.bar_chart(
        history_data.set_index("Item Identifier")["Predicted Sales"]
    )

    # Clear all saved predictions
    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()
else:
    st.info("Your predictions will appear here after you make one.")
