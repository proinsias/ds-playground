import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
import tensorflow as tf
from loguru import logger
from sklearn.model_selection import train_test_split

from disease_risk_prediction.constants import RANDOM_STATE
from disease_risk_prediction.data import fetch_health_data, validate_health_data
from disease_risk_prediction.preprocess import preprocess_training_data
from disease_risk_prediction.train import build_model

# See TensorFlow version.
logger.info(f"TensorFlow version: {tf.__version__}")

# Check for TensorFlow GPU access.
logger.info(
    f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}",
)

# Fetch and preview data
df = fetch_health_data()
df = validate_health_data(df)
logger.info(df.head())

X, y, preprocessor = preprocess_training_data(df)
X_train, y_train, X_test, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# FIXME: Remember to potentially retrain with all data.

# Create a SHAP explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values for the test set
shap_values = explainer(X_test)

# Generate summary plot
st.subheader("Feature Importance Across All Predictions")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Use a subset of training data for SHAP initialization
background = X_train.sample(100, random_state=42)

# Create SHAP DeepExplainer
explainer = shap.DeepExplainer(model, background)

st.title("Disease Risk Prediction")

if st.button("Predict Risk"):
    input_data = np.array(
        [
            st.number_input("Age", min_value=0, max_value=120, value=30),
            st.selectbox("Gender", ["Male", "Female"]),
            st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0),
            st.selectbox("Smoking Status", ["Never", "Former", "Current"]),
            st.selectbox("Alcohol Use", ["Never", "Occasionally", "Regularly"]),
            st.selectbox("Physical Activity", ["Low", "Moderate", "High"]),
            st.selectbox("Blood Pressure", ["Normal", "Elevated", "Hypertension"]),
            st.selectbox("Cholesterol Level", ["Normal", "Borderline", "High"]),
            st.selectbox("Diabetes History", ["No", "Yes"]),
        ],
    ).reshape(1, -1)

    input_data = preprocessor.transform(input_data)

    # Predict risk
    prediction = model.predict(input_data)[0][0]
    risk_percentage = round(prediction * 100, 2)

    # Display result
    st.success(f"Predicted Disease Risk: {risk_percentage}%")

    # Compute SHAP values
    shap_values_single = explainer.shap_values(input_data)

    # Display prediction
    st.success(f"Predicted Disease Risk: {risk_percentage}%")

    # Display SHAP force plot
    st.subheader("Feature Contribution to Prediction")
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values_single[0], max_display=10)
    st.pyplot(fig)

    st.subheader("Explore Feature Impact on Predictions")

    feature = st.selectbox("Select a feature to analyze:", X_train.columns)

    # Plot SHAP dependency plot for selected feature
    fig, ax = plt.subplots()
    shap.dependence_plot(feature, shap_values, X_train, interaction_index=None, ax=ax)
    st.pyplot(fig)
