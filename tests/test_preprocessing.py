"""
Unit tests for data preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import ChurnDataPreprocessor, split_data, get_feature_info


class TestChurnDataPreprocessor(unittest.TestCase):
    """Test cases for ChurnDataPreprocessor class."""
    
    def setUp(self):
        """Set up test data and preprocessor."""
        self.preprocessor = ChurnDataPreprocessor()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'customerID': ['001', '002', '003', '004', '005'],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0, 0, 1],
            'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'Dependents': ['No', 'Yes', 'No', 'Yes', 'No'],
            'tenure': [12, 24, 6, 48, 36],
            'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
            'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes'],
            'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
            'OnlineSecurity': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
            'OnlineBackup': ['Yes', 'No', 'No internet service', 'Yes', 'No'],
            'DeviceProtection': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'TechSupport': ['No', 'No', 'No internet service', 'Yes', 'No'],
            'StreamingTV': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'StreamingMovies': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'One year', 'Month-to-month'],
            'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Credit card (automatic)', 
                            'Bank transfer (automatic)', 'Electronic check'],
            'MonthlyCharges': [50.0, 75.5, 30.0, 60.0, 85.0],
            'TotalCharges': ['600.0', '1811.0', '180.0', '2880.0', '3060.0'],
            'Churn': ['No', 'Yes', 'No', 'No', 'Yes']
        })
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        
        # Check that customerID is removed
        self.assertNotIn('customerID', cleaned_data.columns)
        
        # Check that TotalCharges is converted to numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data['TotalCharges']))
        
        # Check data shape
        self.assertEqual(len(cleaned_data), 5)
    
    def test_standardize_categories(self):
        """Test category standardization."""
        data_with_service_categories = self.sample_data.copy()
        cleaned_data = self.preprocessor._standardize_categories(data_with_service_categories)
        
        # Check that 'No phone service' is replaced with 'No'
        self.assertNotIn('No phone service', cleaned_data['MultipleLines'].values)
        self.assertIn('No', cleaned_data['MultipleLines'].values)
        
        # Check that 'No internet service' is replaced with 'No'
        self.assertNotIn('No internet service', cleaned_data['OnlineSecurity'].values)
    
    def test_encode_features(self):
        """Test feature encoding."""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        encoded_data = self.preprocessor.encode_features(cleaned_data)
        
        # Check that binary features are encoded as 0/1
        binary_features = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for feature in binary_features:
            if feature in encoded_data.columns:
                self.assertTrue(set(encoded_data[feature].unique()).issubset({0, 1}))
        
        # Check that one-hot encoding creates new columns
        self.assertIn('InternetService_DSL', encoded_data.columns)
        self.assertIn('Contract_Month-to-month', encoded_data.columns)
        self.assertIn('PaymentMethod_Electronic check', encoded_data.columns)
    
    def test_scale_features(self):
        """Test feature scaling."""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        encoded_data = self.preprocessor.encode_features(cleaned_data)
        scaled_data = self.preprocessor.scale_features(encoded_data, fit=True)
        
        # Check that scaled features are between 0 and 1
        scaled_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        for feature in scaled_features:
            if feature in scaled_data.columns:
                self.assertTrue(scaled_data[feature].min() >= 0)
                self.assertTrue(scaled_data[feature].max() <= 1)
    
    def test_prepare_features(self):
        """Test feature preparation."""
        cleaned_data = self.preprocessor.clean_data(self.sample_data)
        encoded_data = self.preprocessor.encode_features(cleaned_data)
        scaled_data = self.preprocessor.scale_features(encoded_data, fit=True)
        
        X, y = self.preprocessor.prepare_features(scaled_data, fit=True)
        
        # Check that target is separated correctly
        self.assertNotIn('Churn', X.columns)
        self.assertEqual(len(y), len(X))
        self.assertTrue(set(y.unique()).issubset({0, 1}))
    
    def test_fit_transform(self):
        """Test complete preprocessing pipeline."""
        X, y = self.preprocessor.fit_transform(self.sample_data)
        
        # Check output shapes
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), 5)  # All samples should be preserved
        
        # Check that preprocessor is fitted
        self.assertTrue(self.preprocessor.is_fitted)
        self.assertTrue(len(self.preprocessor.feature_names) > 0)
    
    def test_transform_new_data(self):
        """Test transformation of new data using fitted preprocessor."""
        # First fit the preprocessor
        X_train, y_train = self.preprocessor.fit_transform(self.sample_data)
        
        # Create new test data
        new_data = self.sample_data.iloc[:2].copy()
        
        # Transform new data
        X_test, y_test = self.preprocessor.transform(new_data)
        
        # Check that feature names match
        self.assertEqual(list(X_test.columns), self.preprocessor.feature_names)
        self.assertEqual(len(X_test), 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(np.random.choice([0, 1], 100))
    
    def test_split_data(self):
        """Test data splitting function."""
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, test_size=0.2, random_state=42)
        
        # Check shapes
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Check that splits maintain feature names
        self.assertEqual(list(X_train.columns), list(self.X.columns))
        self.assertEqual(list(X_test.columns), list(self.X.columns))
    
    def test_get_feature_info(self):
        """Test feature information extraction."""
        # Create test data with different feature types
        test_data = pd.DataFrame({
            'numerical': [1.5, 2.3, 3.1, 4.8, 5.2],
            'binary': [0, 1, 1, 0, 1],
            'categorical': ['A', 'B', 'C', 'A', 'B'],
            'high_cardinality': ['X1', 'X2', 'X3', 'X4', 'X5']
        })
        
        feature_info = get_feature_info(test_data)
        
        # Check structure
        self.assertIn('total_features', feature_info)
        self.assertIn('numerical_features', feature_info)
        self.assertIn('binary_features', feature_info)
        self.assertIn('categorical_features', feature_info)
        
        # Check counts
        self.assertEqual(feature_info['total_features'], 4)
        self.assertIn('binary', feature_info['binary_features'])


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation."""
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        invalid_data = pd.DataFrame({
            'tenure': ['invalid', 'string', 'values'],
            'MonthlyCharges': [50.0, 75.0, 'invalid'],
            'Churn': ['Maybe', 'Perhaps', 'No']
        })
        
        preprocessor = ChurnDataPreprocessor()
        
        # This should handle errors gracefully
        try:
            preprocessor.clean_data(invalid_data)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, TypeError))
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        incomplete_data = pd.DataFrame({
            'gender': ['Male', 'Female'],
            'tenure': [12, 24]
            # Missing many required columns
        })
        
        preprocessor = ChurnDataPreprocessor()
        
        # Should handle missing columns gracefully
        try:
            cleaned = preprocessor.clean_data(incomplete_data)
            # At minimum, should not crash
            self.assertIsInstance(cleaned, pd.DataFrame)
        except Exception as e:
            # Should raise appropriate error
            self.assertIsInstance(e, (KeyError, ValueError))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
