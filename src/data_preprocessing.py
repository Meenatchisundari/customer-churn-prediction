"""
Data preprocessing module for customer churn prediction.
Contains functions for cleaning, transforming, and preparing data for modeling.
"""

import logging
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from .config import (
    BINARY_FEATURES, MULTI_CATEGORY_FEATURES, FEATURES_TO_SCALE,
    TARGET_FEATURE, TRAINING_CONFIG, DATA_VALIDATION_RULES
)
from .utils import log_data_info, validate_data

logger = logging.getLogger(__name__)


class ChurnDataPreprocessor:
    """
    Comprehensive data preprocessor for customer churn prediction.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        log_data_info(df, "Raw Data")
        
        df_clean = df.copy()
        
        # Remove customer ID
        if 'customerID' in df_clean.columns:
            df_clean = df_clean.drop('customerID', axis=1)
            logger.info("Removed customerID column")
        
        # Handle TotalCharges column
        if 'TotalCharges' in df_clean.columns:
            # Check for non-numeric values
            non_numeric_mask = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').isnull()
            logger.info(f"Found {non_numeric_mask.sum()} non-numeric values in TotalCharges")
            
            # Remove rows with empty/invalid TotalCharges
            df_clean = df_clean[df_clean['TotalCharges'] != ' ']
            df_clean = df_clean[df_clean['TotalCharges'].notna()]
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            
            # Remove any remaining NaN values
            df_clean = df_clean.dropna(subset=['TotalCharges'])
            
            logger.info(f"Cleaned TotalCharges column, removed {len(df) - len(df_clean)} rows")
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Standardize categorical values
        df_clean = self._standardize_categories(df_clean)
        
        # Validate data
        validation_errors = validate_data(df_clean, DATA_VALIDATION_RULES)
        if validation_errors:
            logger.warning(f"Data validation issues found: {validation_errors}")
        
        log_data_info(df_clean, "Cleaned Data")
        logger.info("Data cleaning completed")
        
        return df_clean
    
    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize categorical values.
        
        Args:
            df: DataFrame to standardize
            
        Returns:
            DataFrame with standardized categories
        """
        df_std = df.copy()
        
        # Replace 'No internet service' and 'No phone service' with 'No'
        df_std = df_std.replace('No internet service', 'No')
        df_std = df_std.replace('No phone service', 'No')
        
        logger.info("Standardized service categories")
        
        return df_std
    
    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with encoded features
        """
        logger.info("Starting feature encoding...")
        
        df_encoded = df.copy()
        
        # Binary encoding for Yes/No columns
        for col in BINARY_FEATURES:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
        
        # Gender encoding
        if 'gender' in df_encoded.columns:
            df_encoded['gender'] = df_encoded['gender'].map({'Female': 1, 'Male': 0})
        
        logger.info("Completed binary encoding")
        
        # One-hot encoding for multi-category columns
        df_encoded = pd.get_dummies(
            df_encoded, 
            columns=MULTI_CATEGORY_FEATURES,
            drop_first=False
        )
        
        logger.info(f"One-hot encoding completed. New shape: {df_encoded.shape}")
        
        # Ensure all columns are numeric
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                logger.warning(f"Column {col} is still object type after encoding")
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame to scale
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Starting feature scaling...")
        
        df_scaled = df.copy()
        
        # Identify columns to scale that exist in the dataframe
        cols_to_scale = [col for col in FEATURES_TO_SCALE if col in df_scaled.columns]
        
        if not cols_to_scale:
            logger.warning("No columns found to scale")
            return df_scaled
        
        if fit:
            df_scaled[cols_to_scale] = self.scaler.fit_transform(df_scaled[cols_to_scale])
            logger.info(f"Fitted scaler and scaled features: {cols_to_scale}")
        else:
            if not hasattr(self.scaler, 'scale_'):
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df_scaled[cols_to_scale] = self.scaler.transform(df_scaled[cols_to_scale])
            logger.info(f"Scaled features using fitted scaler: {cols_to_scale}")
        
        return df_scaled
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = TARGET_FEATURE,
                        fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            fit: Whether to fit preprocessors
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Preparing features for modeling...")
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
            self.is_fitted = True
        
        logger.info(f"Prepared {X.shape[1]} features and {len(y)} target values")
        
        return X, y
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline (fit and transform).
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (processed features, target)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode features
        df_encoded = self.encode_features(df_clean)
        
        # Scale features
        df_scaled = self.scale_features(df_encoded, fit=True)
        
        # Prepare features and target
        X, y = self.prepare_features(df_scaled, fit=True)
        
        logger.info("Preprocessing pipeline completed")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform new data using fitted preprocessors.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (processed features, target)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Encode features
        df_encoded = self.encode_features(df_clean)
        
        # Scale features
        df_scaled = self.scale_features(df_encoded, fit=False)
        
        # Prepare features and target
        X, y = self.prepare_features(df_scaled, fit=False)
        
        # Ensure columns match training data
        missing_cols = set(self.feature_names) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names)
        
        if missing_cols:
            logger.warning(f"Missing columns in new data: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                X[col] = 0
        
        if extra_cols:
            logger.warning(f"Extra columns in new data: {extra_cols}")
            # Remove extra columns
            X = X.drop(columns=extra_cols)
        
        # Reorder columns to match training data
        X = X.reindex(columns=self.feature_names, fill_value=0)
        
        logger.info("Data transformation completed")
        
        return X, y


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = TRAINING_CONFIG['test_size'],
               random_state: int = TRAINING_CONFIG['random_state']) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test data
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Splitting data into train and test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Testing set: {X_test.shape}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    # Log class distribution
    train_distribution = y_train.value_counts(normalize=True)
    test_distribution = y_test.value_counts(normalize=True)
    
    logger.info(f"Training set class distribution: {train_distribution.to_dict()}")
    logger.info(f"Test set class distribution: {test_distribution.to_dict()}")
    
    return X_train, X_test, y_train, y_test


def get_feature_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive information about features.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with feature information
    """
    feature_info = {
        'total_features': len(df.columns),
        'numerical_features': [],
        'categorical_features': [],
        'binary_features': [],
        'feature_types': {},
        'missing_values': {},
        'unique_values': {}
    }
    
    for col in df.columns:
        # Data type
        dtype = str(df[col].dtype)
        feature_info['feature_types'][col] = dtype
        
        # Missing values
        missing_count = df[col].isnull().sum()
        feature_info['missing_values'][col] = missing_count
        
        # Unique values
        unique_count = df[col].nunique()
        feature_info['unique_values'][col] = unique_count
        
        # Categorize feature type
        if dtype in ['int64', 'float64'] and unique_count > 10:
            feature_info['numerical_features'].append(col)
        elif unique_count == 2:
            feature_info['binary_features'].append(col)
        else:
            feature_info['categorical_features'].append(col)
    
    return feature_info


def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of all features.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        DataFrame with feature summary
    """
    summary_data = []
    
    for col in df.columns:
        summary_data.append({
            'feature': col,
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique(),
            'missing_values': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'sample_values': str(df[col].dropna().unique()[:5])
        })
    
    return pd.DataFrame(summary_data)


# Example usage and testing
if __name__ == "__main__":
    # This section can be used for testing the preprocessing pipeline
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example of how to use the preprocessor
    # preprocessor = ChurnDataPreprocessor()
    # df = pd.read_csv("data/raw/your_data.csv")
    # X, y = preprocessor.fit_transform(df)
    # X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Data preprocessing module loaded successfully!")
