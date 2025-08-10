"""
Utility functions for the customer churn prediction project.
Contains helper functions for data processing, visualization, and model evaluation.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve
)
import tensorflow as tf

from .config import PLOT_CONFIG, VISUALIZATIONS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from CSV file with error handling.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        logger.info(f"Data shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def save_data(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path where to save the file
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def save_model_artifacts(model: tf.keras.Model, scaler: Any, 
                        model_path: Union[str, Path],
                        scaler_path: Union[str, Path],
                        metadata: Optional[Dict] = None) -> None:
    """
    Save model and preprocessing artifacts.
    
    Args:
        model: Trained Keras model
        scaler: Fitted scaler object
        model_path: Path to save the model
        scaler_path: Path to save the scaler
        metadata: Additional metadata to save
    """
    try:
        # Save model
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = Path(scaler_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save metadata
        if metadata:
            metadata_path = model_path.parent / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
    except Exception as e:
        logger.error(f"Error saving model artifacts: {e}")
        raise


def load_model_artifacts(model_path: Union[str, Path],
                        scaler_path: Union[str, Path]) -> Tuple[tf.keras.Model, Any]:
    """
    Load model and preprocessing artifacts.
    
    Args:
        model_path: Path to the saved model
        scaler_path: Path to the saved scaler
        
    Returns:
        Tuple of (model, scaler)
    """
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['precision_0'] = report['0']['precision']
    metrics['recall_0'] = report['0']['recall']
    metrics['f1_0'] = report['0']['f1-score']
    metrics['precision_1'] = report['1']['precision']
    metrics['recall_1'] = report['1']['recall']
    metrics['f1_1'] = report['1']['f1-score']
    metrics['macro_avg_precision'] = report['macro avg']['precision']
    metrics['macro_avg_recall'] = report['macro avg']['recall']
    metrics['macro_avg_f1'] = report['macro avg']['f1-score']
    
    # AUC if probabilities provided
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: List[str] = ['No Churn', 'Churn'],
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                   save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                               save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()


def plot_training_history(history: tf.keras.callbacks.History,
                         save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot training history.
    
    Args:
        history: Keras training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray,
                          top_n: int = 15,
                          save_path: Optional[Union[str, Path]] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        feature_names: Names of features
        importance_scores: Importance scores
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    # Create DataFrame and sort by importance
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_importance, x='importance', y='feature', orient='h')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def validate_data(df: pd.DataFrame, validation_rules: Dict) -> List[str]:
    """
    Validate data against predefined rules.
    
    Args:
        df: DataFrame to validate
        validation_rules: Dictionary of validation rules
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for column, rules in validation_rules.items():
        if column not in df.columns:
            errors.append(f"Missing column: {column}")
            continue
            
        if 'min' in rules:
            min_violations = df[df[column] < rules['min']]
            if len(min_violations) > 0:
                errors.append(f"{column}: {len(min_violations)} values below minimum {rules['min']}")
                
        if 'max' in rules:
            max_violations = df[df[column] > rules['max']]
            if len(max_violations) > 0:
                errors.append(f"{column}: {len(max_violations)} values above maximum {rules['max']}")
                
        if 'values' in rules:
            invalid_values = df[~df[column].isin(rules['values'])]
            if len(invalid_values) > 0:
                unique_invalid = invalid_values[column].unique()
                errors.append(f"{column}: Invalid values found: {unique_invalid}")
    
    return errors


def create_model_summary_report(model: tf.keras.Model, metrics: Dict[str, float],
                              feature_importance: Optional[pd.DataFrame] = None,
                              save_path: Optional[Union[str, Path]] = None) -> str:
    """
    Create a comprehensive model summary report.
    
    Args:
        model: Trained Keras model
        metrics: Dictionary of evaluation metrics
        feature_importance: DataFrame with feature importance scores
        save_path: Path to save the report
        
    Returns:
        Report as string
    """
    report_lines = [
        "# Customer Churn Prediction Model Report",
        "=" * 50,
        "",
        "## Model Architecture",
        f"- Total Parameters: {model.count_params():,}",
        f"- Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}",
        f"- Layers: {len(model.layers)}",
        "",
        "## Model Performance",
        f"- Accuracy: {metrics.get('accuracy', 0):.4f}",
        f"- Precision (Class 0): {metrics.get('precision_0', 0):.4f}",
        f"- Recall (Class 0): {metrics.get('recall_0', 0):.4f}",
        f"- F1-Score (Class 0): {metrics.get('f1_0', 0):.4f}",
        f"- Precision (Class 1): {metrics.get('precision_1', 0):.4f}",
        f"- Recall (Class 1): {metrics.get('recall_1', 0):.4f}",
        f"- F1-Score (Class 1): {metrics.get('f1_1', 0):.4f}",
        ""
    ]
    
    if 'roc_auc' in metrics:
        report_lines.extend([
            f"- ROC AUC: {metrics['roc_auc']:.4f}",
            ""
        ])
    
    if feature_importance is not None:
        report_lines.extend([
            "## Top 10 Most Important Features",
            ""
        ])
        for i, row in feature_importance.head(10).iterrows():
            report_lines.append(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
        report_lines.append("")
    
    report_lines.extend([
        "## Model Layers",
        ""
    ])
    
    for i, layer in enumerate(model.layers):
        layer_info = f"{i+1}. {layer.__class__.__name__}"
        if hasattr(layer, 'units'):
            layer_info += f" ({layer.units} units)"
        if hasattr(layer, 'activation'):
            layer_info += f" - {layer.activation.__name__}"
        report_lines.append(layer_info)
    
    report = "\n".join(report_lines)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        logger.info(f"Model report saved to {save_path}")
    
    return report


def setup_directories() -> None:
    """
    Create necessary directories for the project.
    """
    directories = [
        VISUALIZATIONS_DIR,
        Path("data/raw"),
        Path("data/processed"),
        Path("models"),
        Path("logs")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")


def log_data_info(df: pd.DataFrame, stage: str = "Unknown") -> None:
    """
    Log comprehensive information about a DataFrame.
    
    Args:
        df: DataFrame to analyze
        stage: Stage of processing (for logging context)
    """
    logger.info(f"=== Data Info - {stage} ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")
    logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    if 'Churn' in df.columns:
        churn_distribution = df['Churn'].value_counts(normalize=True)
        logger.info(f"Churn distribution: {churn_distribution.to_dict()}")


class ModelTracker:
    """
    Track model experiments and performance.
    """
    
    def __init__(self, log_file: Union[str, Path] = "model_experiments.json"):
        self.log_file = Path(log_file)
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> List[Dict]:
        """Load existing experiments from file."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def log_experiment(self, experiment_name: str, config: Dict, 
                      metrics: Dict, notes: str = "") -> None:
        """
        Log a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration used
            metrics: Performance metrics
            notes: Additional notes
        """
        experiment = {
            "name": experiment_name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "notes": notes
        }
        
        self.experiments.append(experiment)
        
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        logger.info(f"Experiment '{experiment_name}' logged successfully")
    
    def get_best_experiment(self, metric: str = "accuracy") -> Dict:
        """
        Get the best experiment based on a specific metric.
        
        Args:
            metric: Metric to optimize for
            
        Returns:
            Best experiment dictionary
        """
        if not self.experiments:
            return {}
        
        return max(self.experiments, 
                  key=lambda x: x.get("metrics", {}).get(metric, 0))
    
    def compare_experiments(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare all experiments.
        
        Args:
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with experiment comparison
        """
        if not self.experiments:
            return pd.DataFrame()
        
        if metrics is None:
            metrics = ["accuracy", "precision_1", "recall_1", "f1_1"]
        
        comparison_data = []
        for exp in self.experiments:
            row = {"name": exp["name"], "timestamp": exp["timestamp"]}
            for metric in metrics:
                row[metric] = exp.get("metrics", {}).get(metric, None)
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
