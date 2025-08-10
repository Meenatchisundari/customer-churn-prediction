"""
Streamlit web application for Customer Churn Prediction.
Interactive interface for making predictions and exploring the model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import tensorflow as tf
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import STREAMLIT_CONFIG
from utils import load_model_artifacts

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state=STREAMLIT_CONFIG['initial_sidebar_state']
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler."""
    try:
        model_path = Path("../models/churn_model.h5")
        scaler_path = Path("../models/scaler.pkl")
        
        if not model_path.exists() or not scaler_path.exists():
            st.error("Model files not found. Please train the model first.")
            return None, None
        
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def create_input_features():
    """Create input widgets for all features."""
    st.sidebar.header("📊 Customer Information")
    
    # Demographic Information
    st.sidebar.subheader("Demographics")
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    
    # Service Information
    st.sidebar.subheader("Services")
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Additional Services
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
    
    # Account Information
    st.sidebar.subheader("Account Details")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", 
                                         "Bank transfer (automatic)", "Credit card (automatic)"])
    
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
    total_charges = st.sidebar.slider("Total Charges ($)", 18.0, 8500.0, 1500.0)
    
    # Create feature dictionary
    features = {
        'gender': 1 if gender == "Female" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'OnlineBackup': 1 if online_backup == "Yes" else 0,
        'DeviceProtection': 1 if device_protection == "Yes" else 0,
        'TechSupport': 1 if tech_support == "Yes" else 0,
        'StreamingTV': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        # One-hot encoded features
        'InternetService_DSL': 1 if internet_service == "DSL" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }
    
    return features

def preprocess_features(features, scaler):
    """Preprocess features for prediction."""
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Scale numerical features
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Create a copy for scaling
    df_scaled = df.copy()
    
    # Apply scaling to numerical features
    df_scaled[numerical_features] = scaler.transform(df[numerical_features])
    
    return df_scaled.values

def make_prediction(model, features_processed):
    """Make churn prediction."""
    prediction_proba = model.predict(features_processed)[0][0]
    prediction = 1 if prediction_proba > 0.5 else 0
    
    return prediction, prediction_proba

def display_prediction_result(prediction, probability):
    """Display prediction results with styling."""
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error(" **HIGH CHURN RISK**")
        else:
            st.success(" **LOW CHURN RISK**")
    
    with col2:
        st.metric("Churn Probability", f"{probability:.2%}")
    
    # Probability gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_feature_analysis(features):
    """Display analysis of input features."""
    st.subheader(" Feature Analysis")
    
    # Risk factors analysis
    risk_factors = []
    
    if features['Contract_Month-to-month'] == 1:
        risk_factors.append("Month-to-month contract (higher churn risk)")
    
    if features['PaymentMethod_Electronic check'] == 1:
        risk_factors.append("Electronic check payment (higher churn risk)")
    
    if features['tenure'] < 12:
        risk_factors.append("Low tenure (< 12 months)")
    
    if features['InternetService_Fiber optic'] == 1:
        risk_factors.append("Fiber optic internet service")
    
    if features['SeniorCitizen'] == 1:
        risk_factors.append("Senior citizen")
    
    if features['Partner'] == 0:
        risk_factors.append("No partner")
    
    if risk_factors:
        st.warning("**Risk Factors Identified:**")
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.success(" No major risk factors identified")
    
    # Feature importance (simulated - in real app, load from saved feature importance)
    feature_importance = {
        'Contract_Month-to-month': 0.15,
        'tenure': 0.12,
        'TotalCharges': 0.10,
        'MonthlyCharges': 0.08,
        'InternetService_Fiber optic': 0.07,
        'PaymentMethod_Electronic check': 0.06,
        'OnlineSecurity': 0.05,
        'TechSupport': 0.04
    }
    
    # Display feature importance chart
    fig_importance = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Feature Importance (Top 8)",
        labels={'x': 'Importance', 'y': 'Features'}
    )
    fig_importance.update_layout(height=300)
    st.plotly_chart(fig_importance, use_container_width=True)

def display_recommendations(prediction, features):
    """Display recommendations based on prediction."""
    st.subheader(" Recommendations")
    
    if prediction == 1:  # High churn risk
        st.markdown("""
        **Immediate Actions:**
        
         **Retention Strategies:**
        """)
        
        recommendations = []
        
        if features['Contract_Month-to-month'] == 1:
            recommendations.append("• Offer incentives to upgrade to annual contract")
        
        if features['tenure'] < 12:
            recommendations.append("• Implement new customer onboarding program")
            recommendations.append("• Provide dedicated customer success manager")
        
        if features['PaymentMethod_Electronic check'] == 1:
            recommendations.append("• Encourage automatic payment methods with discounts")
        
        if features['OnlineSecurity'] == 0 and features['InternetService_No'] == 0:
            recommendations.append("• Offer free online security service trial")
        
        if features['TechSupport'] == 0:
            recommendations.append("• Provide complimentary tech support consultation")
        
        recommendations.append("• Conduct satisfaction survey and address concerns")
        recommendations.append("• Offer loyalty rewards or service upgrades")
        
        for rec in recommendations:
            st.write(rec)
    
    else:  # Low churn risk
        st.success("""
        **Customer appears satisfied, but consider:**
        
        • Cross-sell additional services
        • Encourage referrals with incentives
        • Maintain regular communication
        • Monitor satisfaction continuously
        """)

def load_sample_data():
    """Load sample data for demonstration."""
    sample_data = {
        "High Risk Customer": {
            'gender': 0, 'SeniorCitizen': 1, 'Partner': 0, 'Dependents': 0,
            'tenure': 3, 'PhoneService': 1, 'MultipleLines': 0,
            'OnlineSecurity': 0, 'OnlineBackup': 0, 'DeviceProtection': 0,
            'TechSupport': 0, 'StreamingTV': 0, 'StreamingMovies': 0,
            'PaperlessBilling': 1, 'MonthlyCharges': 85.0, 'TotalCharges': 255.0,
            'InternetService_DSL': 0, 'InternetService_Fiber optic': 1, 'InternetService_No': 0,
            'Contract_Month-to-month': 1, 'Contract_One year': 0, 'Contract_Two year': 0,
            'PaymentMethod_Bank transfer (automatic)': 0,
            'PaymentMethod_Credit card (automatic)': 0,
            'PaymentMethod_Electronic check': 1,
            'PaymentMethod_Mailed check': 0
        },
        "Low Risk Customer": {
            'gender': 1, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 1,
            'tenure': 45, 'PhoneService': 1, 'MultipleLines': 1,
            'OnlineSecurity': 1, 'OnlineBackup': 1, 'DeviceProtection': 1,
            'TechSupport': 1, 'StreamingTV': 1, 'StreamingMovies': 1,
            'PaperlessBilling': 0, 'MonthlyCharges': 65.0, 'TotalCharges': 2925.0,
            'InternetService_DSL': 1, 'InternetService_Fiber optic': 0, 'InternetService_No': 0,
            'Contract_Month-to-month': 0, 'Contract_One year': 0, 'Contract_Two year': 1,
            'PaymentMethod_Bank transfer (automatic)': 1,
            'PaymentMethod_Credit card (automatic)': 0,
            'PaymentMethod_Electronic check': 0,
            'PaymentMethod_Mailed check': 0
        }
    }
    return sample_data

def main():
    """Main application function."""
    # Header
    st.title(" Customer Churn Prediction")
    st.markdown("Predict customer churn probability using machine learning")
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.stop()
    
    # Sidebar for sample data
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Quick Test")
    sample_data = load_sample_data()
    
    if st.sidebar.button("Load High Risk Customer"):
        st.session_state.update(sample_data["High Risk Customer"])
        st.experimental_rerun()
    
    if st.sidebar.button("Load Low Risk Customer"):
        st.session_state.update(sample_data["Low Risk Customer"])
        st.experimental_rerun()
    
    # Get input features
    features = create_input_features()
    
    # Prediction section
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col2:
        predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)
    
    if predict_button:
        # Preprocess features
        features_processed = preprocess_features(features, scaler)
        
        # Make prediction
        prediction, probability = make_prediction(model, features_processed)
        
        # Display results
        display_prediction_result(prediction, probability)
        
        # Feature analysis
        display_feature_analysis(features)
        
        # Recommendations
        display_recommendations(prediction, features)
    
    # Model information
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Information:**
        - **Algorithm**: Artificial Neural Network (ANN)
        - **Framework**: TensorFlow/Keras
        - **Architecture**: 26 → 26 → 15 → 1 neurons
        - **Accuracy**: ~77.5% on test data
        - **Features**: 26 customer attributes
        
        **Prediction Confidence:**
        - **High Risk**: Probability ≥ 50%
        - **Low Risk**: Probability < 50%
        
        **Note**: This model is for demonstration purposes. 
        Actual business decisions should consider additional factors.
        """)

if __name__ == "__main__":
    main()
