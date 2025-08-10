"""
Unit tests for model module.
"""

import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import ChurnPredictionModel, ModelEnsemble, create_baseline_model
from config import MODEL_CONFIG


class TestChurnPredictionModel(unittest.TestCase):
    """Test cases for ChurnPredictionModel class."""
    
    def setUp(self):
        """Set up test data and model."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        self.model = ChurnPredictionModel()
        
        # Create sample training data
        self.n_samples = 100
        self.n_features = 26
        self.X_train = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_train = pd.Series(np.random.choice([0, 1], self.n_samples))
        
        # Create sample test data
        self.X_test = pd.DataFrame(
            np.random.randn(20, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_test = pd.Series(np.random.choice([0, 1], 20))
    
    def test_model_initialization(self):
        """Test model initialization."""
        # Test default initialization
        model = ChurnPredictionModel()
        self.assertIsNotNone(model.config)
        self.assertFalse(model.is_trained)
        self.assertIsNone(model.model)
        
        # Test custom config initialization
        custom_config = MODEL_CONFIG.copy()
        custom_config['hidden_layers'] = [32, 16]
        model_custom = ChurnPredictionModel(custom_config)
        self.assertEqual(model_custom.config['hidden_layers'], [32, 16])
    
    def test_build_model(self):
        """Test model building."""
        model = self.model.build_model(self.n_features)
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertIsInstance(model, tf.keras.Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, self.n_features))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check that model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_model_training(self):
        """Test model training."""
        # Build model
        self.model.build_model(self.n_features)
        
        # Train model (short training for testing)
        history = self.model.train(
            self.X_train, self.y_train,
            epochs=2, 
            batch_size=32,
            verbose=0
        )
        
        # Check training results
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
        
        # Check feature names are stored
        self.assertEqual(self.model.feature_names, list(self.X_train.columns))
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Build and train model
        self.model.build_model(self.n_features)
        self.model.train(
            self.X_train, self.y_train,
            epochs=2,
            verbose=0
        )
        
        # Make predictions
        y_pred, y_pred_proba = self.model.predict(self.X_test)
        
        # Check prediction shapes
        self.assertEqual(len(y_pred), len(self.X_test))
        self.assertEqual(len(y_pred_proba), len(self.X_test))
        
        # Check prediction values
        self.assertTrue(all(pred in [0, 1] for pred in y_pred))
        self.assertTrue(all(0 <= prob <= 1 for prob in y_pred_proba))
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Build and train model
        self.model.build_model(self.n_features)
        self.model.train(
            self.X_train, self.y_train,
            epochs=2,
            verbose=0
        )
        
        # Evaluate model
        metrics = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        
        # Check metrics structure
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_0', metrics)
        self.assertIn('recall_0', metrics)
        self.assertIn('f1_0', metrics)
        
        # Check metric values are reasonable
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision_0'] <= 1)
    
    def test_model_validation(self):
        """Test model performance validation."""
        # Create mock metrics
        good_metrics = {
            'accuracy': 0.80,
            'precision_1': 0.65,
            'recall_1': 0.60,
            'f1_1': 0.62
        }
        
        bad_metrics = {
            'accuracy': 0.70,
            'precision_1': 0.50,
            'recall_1': 0.45,
            'f1_1': 0.47
        }
        
        # Test validation
        self.assertTrue(self.model.validate_performance(good_metrics))
        self.assertFalse(self.model.validate_performance(bad_metrics))
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Build and train model
        self.model.build_model(self.n_features)
        self.model.train(
            self.X_train, self.y_train,
            epochs=2,
            verbose=0
        )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save model
            self.model.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create new model instance and load
            new_model = ChurnPredictionModel()
            new_model.load_model(temp_path)
            
            # Check that model is loaded
            self.assertTrue(new_model.is_trained)
            self.assertIsNotNone(new_model.model)
            
            # Test predictions are consistent
            pred1, _ = self.model.predict(self.X_test)
            pred2, _ = new_model.predict(self.X_test)
            np.testing.assert_array_equal(pred1, pred2)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training."""
        with self.assertRaises(ValueError):
            self.model.predict(self.X_test)
    
    def test_evaluation_without_training(self):
        """Test that evaluation fails without training."""
        with self.assertRaises(ValueError):
            self.model.evaluate(self.X_test, self.y_test)


class TestModelEnsemble(unittest.TestCase):
    """Test cases for ModelEnsemble class."""
    
    def setUp(self):
        """Set up test data and ensemble."""
        # Set random seeds
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create sample data
        self.n_samples = 50  # Smaller for faster testing
        self.n_features = 26
        self.X_train = pd.DataFrame(
            np.random.randn(self.n_samples, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_train = pd.Series(np.random.choice([0, 1], self.n_samples))
        
        self.X_test = pd.DataFrame(
            np.random.randn(10, self.n_features),
            columns=[f'feature_{i}' for i in range(self.n_features)]
        )
        self.y_test = pd.Series(np.random.choice([0, 1], 10))
        
        # Create ensemble with 2 models for faster testing
        configs = [
            {**MODEL_CONFIG, 'hidden_layers': [16, 8]},
            {**MODEL_CONFIG, 'hidden_layers': [20, 10]}
        ]
        self.ensemble = ModelEnsemble(configs)
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        self.assertEqual(len(self.ensemble.models), 2)
        self.assertFalse(self.ensemble.is_trained)
        
        # Test each model is correctly initialized
        for model in self.ensemble.models:
            self.assertIsInstance(model, ChurnPredictionModel)
    
    def test_ensemble_training(self):
        """Test ensemble training."""
        histories = self.ensemble.train_all(
            self.X_train, self.y_train,
            epochs=2,
            verbose=0
        )
        
        # Check training results
        self.assertTrue(self.ensemble.is_trained)
        self.assertEqual(len(histories), 2)
        
        # Check that all models are trained
        for model in self.ensemble.models:
            self.assertTrue(model.is_trained)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        # Train ensemble
        self.ensemble.train_all(
            self.X_train, self.y_train,
            epochs=2,
            verbose=0
        )
        
        # Test different ensemble methods
        for method in ['average', 'majority', 'weighted']:
            pred, prob = self.ensemble.predict(self.X_test, method=method)
            
            # Check shapes
            self.assertEqual(len(pred), len(self.X_test))
            self.assertEqual(len(prob), len(self.X_test))
            
            # Check values
            self.assertTrue(all(p in [0, 1] for p in pred))
            self.assertTrue(all(0 <= p <= 1 for p in prob))


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_baseline_model(self):
        """Test baseline model creation."""
        input_dim = 26
        model = create_baseline_model(input_dim)
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, input_dim))
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check that model is compiled
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)


class TestModelConfiguration(unittest.TestCase):
    """Test cases for model configuration."""
    
    def test_model_config_validation(self):
        """Test that model configuration is valid."""
        # Check required keys exist
        required_keys = ['input_dim', 'hidden_layers', 'dropout_rate', 
                        'activation', 'output_activation', 'optimizer', 'loss']
        
        for key in required_keys:
            self.assertIn(key, MODEL_CONFIG)
        
        # Check value types
        self.assertIsInstance(MODEL_CONFIG['hidden_layers'], list)
        self.assertIsInstance(MODEL_CONFIG['dropout_rate'], (int, float))
        self.assertTrue(0 <= MODEL_CONFIG['dropout_rate'] <= 1)
    
    def test_custom_config(self):
        """Test custom configuration handling."""
        custom_config = {
            **MODEL_CONFIG,
            'hidden_layers': [64, 32, 16],
            'dropout_rate': 0.3
        }
        
        model = ChurnPredictionModel(custom_config)
        self.assertEqual(model.config['hidden_layers'], [64, 32, 16])
        self.assertEqual(model.config['dropout_rate'], 0.3)


if __name__ == '__main__':
    # Set TensorFlow to run quietly during tests
    tf.get_logger().setLevel('ERROR')
    
    # Run tests
    unittest.main(verbosity=2)
