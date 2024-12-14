import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import plotly.express as px

# Load data and models
@st.cache_data
def load_data():
    data_path = "churn-bigml-80.csv"
    data = pd.read_csv(data_path)
    return data

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# App Layout
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Choose Section", ["Overview", "Data Visualization", "Model Prediction"]
)


# Overview Section
if option == "Overview":
    st.header("Overview")
    st.markdown(
        """
        Welcome to the Customer Churn Prediction Dashboard! 

        ### Key Features:
        - **Data Visualization**: Explore the dataset through interactive visualizations.
        - **Model Selection**: Choose from pre-trained models for predictions.
        - **Real-Time Predictions**: Predict customer churn based on input features.

        ### Objective:
        This project analyzes customer churn to help businesses retain customers and reduce revenue loss.
        """
    )

# Data Visualization Section
elif option == "Data Visualization":
    st.header("Data Visualization")
    data = load_data()

    st.subheader("Dataset Overview")
    st.write("First 10 rows of the dataset:")
    st.dataframe(data.head(10))

    st.subheader("Statistical Summary")
    st.write(data.describe())

    st.subheader("Distribution of Churn")
    churn_distribution = data["Churn"].value_counts().reset_index()
    churn_distribution.columns = ["Churn", "Count"]
    fig_pie = px.pie(
        churn_distribution,
        names="Churn",
        values="Count",
        title="Churn Distribution",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig_pie)

    st.subheader("Histogram of Features")
    selected_feature = st.selectbox("Choose a feature to plot:", data.columns)
    if selected_feature:
        fig_hist = px.histogram(
            data,
            x=selected_feature,
            color="Churn",
            barmode="overlay",
            title=f"Histogram of {selected_feature} by Churn",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_hist)



# Model Prediction Section
elif option == "Model Prediction":
    st.header("Make a Prediction")

    # Load available models
    models = {
        "Random Forest": "models/rf_model.pkl",
        "KNN": "models/knn_model.pkl",
        "Gaussian Naive Bayes": "models/gnb_model.pkl",
        "SVM (Linear Kernel)": "models/svm_linear_model.pkl",
        "SVM (RBF Kernel)": "models/svm_rbf_model.pkl",
        "LightGBM": "models/lgb_model.pkl",
        "XGBoost": "models/xgb_model.pkl",
    }

    # Select a model
    selected_model = st.selectbox("Choose a Model", list(models.keys()))
    model_path = models[selected_model]
    model = load_model(model_path)

    # Get the feature names dynamically
    try:
        required_features = model.feature_names_in_
    except AttributeError:
        required_features = None  # Handle models that don't have this attribute

    # Create input fields dynamically based on the required features
    st.subheader("Input Customer Features")
    input_data = {}

    if required_features is not None:
        for feature in required_features:
            if feature in ["State", "Area code", "International plan", "Voice mail plan"]:
                # Categorical features with example inputs
                if feature == "State":
                    input_data[feature] = st.text_input(f"{feature}", "CA")
                elif feature == "Area code":
                    input_data[feature] = st.number_input(f"{feature}", value=415, step=1)
                elif feature in ["International plan", "Voice mail plan"]:
                    input_data[feature] = 1 if st.selectbox(
                        f"{feature}", ["Yes", "No"]
                    ).lower() == "yes" else 0
            else:
                # Numerical features
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    else:
        st.warning("Model does not provide feature names dynamically.")

    # Convert categorical strings to numeric values
    if "State" in input_data:
        input_data["State"] = 0  # Example: Encode all "State" as 0

    # Convert the input_data dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make predictions
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Results")
        st.write(f"Prediction: **{'Churn' if prediction[0] == 1 else 'Not Churn'}**")
        st.write(f"Probability of Churn: **{prediction_proba[0][1] * 100:.2f}%**")

        st.subheader("Selected Model")
        st.write(f"Using: **{selected_model}**")

        st.success("Prediction complete!")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.write(
    """
    This dashboard was created as part of a customer churn prediction project.
    """
)
