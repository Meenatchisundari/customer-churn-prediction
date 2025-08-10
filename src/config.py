"""
Configuration file for the customer churn prediction project.
Contains all the constants and configuration parameters.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file paths
RAW_DATA_FILE = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_churn_data.csv"

# Model file paths
MODEL_FILE = MODELS_DIR / "churn_model.h5"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
ENCODER_FILE = MODELS_DIR / "encoder.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Model parameters
MODEL_CONFIG = {
    "input_dim": 26,
    "hidden_layers": [26, 15],
    "dropout_rate": 0.2,
    "activation": "relu",
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy", "precision", "recall"]
}

# Training parameters
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.2
}

# Feature configuration
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET_FEATURE = "Churn"

FEATURES_TO_SCALE = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling"
]

MULTI_CATEGORY_FEATURES = ["InternetService", "Contract", "PaymentMethod"]

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    "handle_missing_values": True,
    "remove_duplicates": True,
    "standardize_categories": True,
    "scale_features": True,
    "encode_categorical": True
}

# Visualization parameters
PLOT_CONFIG = {
    "figure_size": (12, 8),
    "color_palette": "husl",
    "style": "whitegrid",
    "dpi": 300,
    "save_format": "png"
}

# API configuration (for deployment)
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "reload": False
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Customer Churn Prediction",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False
        }
    }
}

# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.75,
    "min_precision": 0.60,
    "min_recall": 0.55,
    "min_f1_score": 0.60
}

# Feature importance threshold
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Data validation rules
DATA_VALIDATION_RULES = {
    "tenure": {"min": 0, "max": 100},
    "MonthlyCharges": {"min": 0, "max": 200},
    "TotalCharges": {"min": 0, "max": 10000},
    "SeniorCitizen": {"values": [0, 1]},
    "gender": {"values": ["Male", "Female"]},
    "Contract": {"values": ["Month-to-month", "One year", "Two year"]},
    "InternetService": {"values": ["DSL", "Fiber optic", "No"]},
    "PaymentMethod": {"values": [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]}
}
