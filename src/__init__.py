"""
Customer Churn Prediction Package

A comprehensive machine learning package for predicting customer churn
in the telecommunications industry using deep learning techniques.

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .config import MODEL_CONFIG, TRAINING_CONFIG
from .data_preprocessing import ChurnDataPreprocessor, split_data
from .model import ChurnPredictionModel, ModelEnsemble
from .utils import (
    load_data,
    save_data,
    calculate_metrics,
    plot_confusion_matrix,
    plot_roc_curve
)

__all__ = [
    "MODEL_CONFIG",
    "TRAINING_CONFIG", 
    "ChurnDataPreprocessor",
    "split_data",
    "ChurnPredictionModel",
    "ModelEnsemble",
    "load_data",
    "save_data",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve"
]
