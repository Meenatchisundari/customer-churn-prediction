"""
Model building and training module for customer churn prediction.
Contains the neural network architecture and training utilities.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import classification_report

from .config import MODEL_CONFIG, TRAINING_CONFIG, PERFORMANCE_THRESHOLDS
from .utils import calculate_metrics, plot_training_history, ModelTracker

logger = logging.getLogger(__name__)


class ChurnPredictionModel:
    """
    Customer churn prediction model using Artificial Neural Networks.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the model with configuration.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or MODEL_CONFIG.copy()
        self.model = None
        self.history = None
        self.is_trained = False
        self.feature_names = []
        
    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras model
        """
        logger.info("Building neural network model...")
        
        model = keras.Sequential([
            # Input layer
            layers.Dense(
                self.config['hidden_layers'][0],
                input_shape=(input_dim,),
                activation=self.config['activation'],
                name='input_layer'
            ),
            layers.Dropout(self.config['dropout_rate'], name='dropout_1'),
            
            # Hidden layer
            layers.Dense(
                self.config['hidden_layers'][1],
                activation=self.config['activation'],
                name='hidden_layer'
            ),
            layers.Dropout(self.config['dropout_rate'], name='dropout_2'),
            
            # Output layer
            layers.Dense(
                1,
                activation=self.config['output_activation'],
                name='output_layer'
            )
        ])
        
        # Compile model
        model.compile(
            optimizer=self.config['optimizer'],
            loss=self.config['loss'],
            metrics=self.config['metrics']
        )
        
        self.model = model
        
        logger.info("Model compiled successfully")
        logger.info(f"Model parameters: {model.count_params():,}")
        
        return model
    
    def create_callbacks(self) -> List[callbacks.Callback]:
        """
        Create training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING_CONFIG['reduce_lr_factor'],
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint (optional)
        # checkpoint = callbacks.ModelCheckpoint(
        #     'best_model.h5',
        #     monitor='val_accuracy',
        #     save_best_only=True,
        #     verbose=1
        # )
        # callback_list.append(checkpoint)
        
        return callback_list
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              epochs: int = None, batch_size: int = None,
              verbose: int = 1) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Set default parameters
        epochs = epochs or TRAINING_CONFIG['epochs']
        batch_size = batch_size or TRAINING_CONFIG['batch_size']
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            # Use validation split
            validation_split = TRAINING_CONFIG.get('validation_split', 0.2)
            logger.info(f"Using validation split: {validation_split}")
        
        # Create callbacks
        callback_list = self.create_callbacks()
        
        logger.info("Starting model training...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return self.history
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            threshold: Decision threshold for binary classification
            
        Returns:
            Tuple of (predicted_classes, predicted_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get probability predictions
        y_pred_proba = self.model.predict(X)
        
        # Convert to binary predictions
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        
        return y_pred, y_pred_proba.flatten()
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 verbose: int = 1) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Verbosity level
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model performance...")
        
        # Make predictions
        y_pred, y_pred_proba = self.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Log results
        if verbose:
            logger.info("Model Performance:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision (Churn): {metrics['precision_1']:.4f}")
            logger.info(f"Recall (Churn): {metrics['recall_1']:.4f}")
            logger.info(f"F1-Score (Churn): {metrics['f1_1']:.4f}")
            if 'roc_auc' in metrics:
                logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def validate_performance(self, metrics: Dict[str, float]) -> bool:
        """
        Validate if model meets performance thresholds.
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            True if model meets all thresholds
        """
        validation_results = {}
        
        for metric, threshold in PERFORMANCE_THRESHOLDS.items():
            actual_value = metrics.get(metric, 0)
            validation_results[metric] = actual_value >= threshold
            
            if validation_results[metric]:
                logger.info(f"✓ {metric}: {actual_value:.4f} >= {threshold}")
            else:
                logger.warning(f"✗ {metric}: {actual_value:.4f} < {threshold}")
        
        all_passed = all(validation_results.values())
        
        if all_passed:
            logger.info("Model meets all performance thresholds")
        else:
            logger.warning("Model does not meet all performance thresholds")
        
        return all_passed
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model not built yet."
        
        summary_str = []
        self.model.summary(print_fn=summary_str.append)
        return '\n'.join(summary_str)
    
    def save_model(self, model_path: str, save_weights_only: bool = False) -> None:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
            save_weights_only: Whether to save only weights
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        if save_weights_only:
            self.model.save_weights(model_path)
            logger.info(f"Model weights saved to {model_path}")
        else:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, load_weights_only: bool = False) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            load_weights_only: Whether to load only weights
        """
        if load_weights_only:
            if self.model is None:
                raise ValueError("Model architecture not built. Cannot load weights.")
            self.model.load_weights(model_path)
            logger.info(f"Model weights loaded from {model_path}")
        else:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        
        self.is_trained = True


class ModelEnsemble:
    """
    Ensemble of multiple churn prediction models for improved performance.
    """
    
    def __init__(self, model_configs: List[Dict] = None):
        """
        Initialize ensemble with multiple model configurations.
        
        Args:
            model_configs: List of configuration dictionaries for each model
        """
        if model_configs is None:
            # Default ensemble with different architectures
            model_configs = [
                {**MODEL_CONFIG, 'hidden_layers': [32, 16]},
                {**MODEL_CONFIG, 'hidden_layers': [26, 15]},
                {**MODEL_CONFIG, 'hidden_layers': [20, 10]}
            ]
        
        self.models = [ChurnPredictionModel(config) for config in model_configs]
        self.is_trained = False
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame = None, y_val: pd.Series = None,
                  **training_kwargs) -> List[keras.callbacks.History]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **training_kwargs: Additional training arguments
            
        Returns:
            List of training histories
        """
        logger.info(f"Training ensemble of {len(self.models)} models...")
        
        histories = []
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}")
            history = model.train(X_train, y_train, X_val, y_val, **training_kwargs)
            histories.append(history)
        
        self.is_trained = True
        logger.info("Ensemble training completed")
        
        return histories
    
    def predict(self, X: pd.DataFrame, method: str = 'average') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Features to predict
            method: Ensemble method ('average', 'majority', 'weighted')
            
        Returns:
            Tuple of (predicted_classes, predicted_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Ensemble not trained. Call train_all() first.")
        
        # Get predictions from all models
        all_predictions = []
        all_probabilities = []
        
        for model in self.models:
            pred, prob = model.predict(X)
            all_predictions.append(pred)
            all_probabilities.append(prob)
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        # Combine predictions based on method
        if method == 'average':
            ensemble_prob = np.mean(all_probabilities, axis=0)
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
        elif method == 'majority':
            ensemble_pred = np.round(np.mean(all_predictions, axis=0)).astype(int)
            ensemble_prob = np.mean(all_probabilities, axis=0)
        elif method == 'weighted':
            # Simple equal weighting - could be improved with performance-based weights
            weights = np.ones(len(self.models)) / len(self.models)
            ensemble_prob = np.average(all_probabilities, axis=0, weights=weights)
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred, ensemble_prob


def create_baseline_model(input_dim: int) -> keras.Model:
    """
    Create a simple baseline model for comparison.
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled baseline model
    """
    model = keras.Sequential([
        layers.Dense(10, input_shape=(input_dim,), activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def hyperparameter_search(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         param_grid: Dict = None) -> Dict:
    """
    Perform simple grid search for hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        param_grid: Grid of parameters to search
        
    Returns:
        Best parameters and results
    """
    if param_grid is None:
        param_grid = {
            'hidden_layers': [[32, 16], [26, 15], [20, 10], [50, 25]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.1]
        }
    
    logger.info("Starting hyperparameter search...")
    
    best_score = 0
    best_params = {}
    results = []
    
    # Simple grid search implementation
    for hidden_layers in param_grid['hidden_layers']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                
                config = MODEL_CONFIG.copy()
                config['hidden_layers'] = hidden_layers
                config['dropout_rate'] = dropout_rate
                
                # Create and train model
                model = ChurnPredictionModel(config)
                model.build_model(X_train.shape[1])
                
                # Compile with specific learning rate
                model.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                history = model.train(
                    X_train, y_train, X_val, y_val,
                    epochs=50, verbose=0
                )
                
                # Evaluate
                metrics = model.evaluate(X_val, y_val, verbose=0)
                score = metrics['accuracy']
                
                params = {
                    'hidden_layers': hidden_layers,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate
                }
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                logger.info(f"Params: {params}, Score: {score:.4f}")
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


def train_with_cross_validation(X: pd.DataFrame, y: pd.Series,
                               n_folds: int = 5) -> Dict:
    """
    Train model with cross-validation.
    
    Args:
        X: Features
        y: Targets
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold
    
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    cv_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        model = ChurnPredictionModel()
        model.train(X_train_fold, y_train_fold, verbose=0)
        
        # Evaluate
        metrics = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(metrics['accuracy'])
        cv_metrics.append(metrics)
    
    cv_results = {
        'cv_scores': cv_scores,
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'cv_metrics': cv_metrics
    }
    
    logger.info(f"CV Score: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    return cv_results


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example of how to use the model
    # model = ChurnPredictionModel()
    # model.build_model(input_dim=26)
    # history = model.train(X_train, y_train, X_val, y_val)
    # metrics = model.evaluate(X_test, y_test)
    
    print("Model module loaded successfully!")
