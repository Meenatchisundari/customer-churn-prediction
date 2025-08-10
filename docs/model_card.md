#  Model Card: Customer Churn Prediction

## Model Overview

| Attribute | Value |
|-----------|--------|
| **Model Name** | Customer Churn Prediction ANN |
| **Model Type** | Artificial Neural Network (Binary Classification) |
| **Framework** | TensorFlow/Keras |
| **Version** | 1.0.0 |
| **Date Created** | 2024 |
| **Last Updated** | 2024 |

## Intended Use

### Primary Use Case
Predict the likelihood of customer churn in the telecommunications industry to enable proactive retention strategies.

### Intended Users
- **Business analysts** for customer segmentation and retention planning
- **Marketing teams** for targeted retention campaigns
- **Customer success managers** for identifying at-risk customers
- **Data scientists** for model monitoring and improvement

### Out-of-Scope Uses
-  **Real-time fraud detection**
-  **Credit scoring or financial risk assessment**
-  **Healthcare or medical predictions**
-  **Cross-industry churn prediction without retraining**

## Model Architecture

### Network Structure
```
Input Layer:    26 neurons (ReLU activation)
Hidden Layer:   15 neurons (ReLU activation)
Output Layer:   1 neuron (Sigmoid activation)

Total Parameters: 831
Trainable Parameters: 831
```

### Key Components
- **Dropout Layers**: 20% dropout rate for regularization
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Optimizer**: Adam with default learning rate (0.001)
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall

## Training Data

### Dataset Information
- **Source**: IBM Watson Analytics Sample Dataset
- **Size**: 7,043 customer records
- **Features**: 20 original features → 26 after preprocessing
- **Target Distribution**: 73% No Churn, 27% Churn
- **Split**: 80% training, 20% testing
- **Validation**: Stratified sampling to maintain class distribution

### Data Preprocessing
1. **Cleaning**: Removed missing values and duplicates
2. **Encoding**: Binary and one-hot encoding for categorical variables
3. **Scaling**: MinMax scaling for numerical features
4. **Validation**: Data quality checks and validation rules

## Performance Metrics

### Overall Performance
| Metric | Value |
|--------|--------|
| **Accuracy** | 77.5% |
| **Precision (No Churn)** | 83% |
| **Recall (No Churn)** | 86% |
| **F1-Score (No Churn)** | 84% |
| **Precision (Churn)** | 62% |
| **Recall (Churn)** | 56% |
| **F1-Score (Churn)** | 59% |
| **ROC AUC** | 0.78 |

### Performance by Class
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **No Churn (0)** | 0.83 | 0.86 | 0.84 | 999 |
| **Churn (1)** | 0.62 | 0.56 | 0.59 | 408 |
| **Macro Avg** | 0.73 | 0.71 | 0.72 | 1407 |
| **Weighted Avg** | 0.77 | 0.77 | 0.77 | 1407 |

### Business Metrics
- **False Positive Rate**: 14% (customers incorrectly flagged as churn risk)
- **False Negative Rate**: 44% (churning customers not identified)
- **Cost-Benefit Analysis**: Model saves ~$2,500 per correctly identified churning customer

## Limitations and Considerations

### Model Limitations
1. **Class Imbalance**: Lower performance on minority class (churn)
2. **Feature Dependencies**: Some features may be correlated
3. **Temporal Aspects**: Model doesn't capture seasonal patterns
4. **External Factors**: Doesn't account for market competition or economic factors

### Biases and Fairness
- **Demographic Bias**: Performance may vary across different customer segments
- **Historical Bias**: Model reflects historical patterns that may not apply to future scenarios
- **Selection Bias**: Dataset may not represent all customer types

### Uncertainty and Confidence
- **Prediction Confidence**: Model provides probability scores (0-1)
- **Decision Threshold**: 0.5 threshold optimized for balanced accuracy
- **Calibration**: Probability scores are reasonably well-calibrated

## Ethical Considerations

### Privacy and Data Protection
- **Data Anonymization**: Customer IDs removed before training
- **GDPR Compliance**: Model design follows data protection principles
- **Data Retention**: Clear policies for model data lifecycle

### Fairness and Non-discrimination
- **Protected Attributes**: Gender and age (senior citizen) monitored for bias
- **Equal Treatment**: Model performance evaluated across demographic groups
- **Transparency**: Model decisions can be explained through feature importance

### Potential Risks
- **Customer Experience**: False positives may lead to unnecessary retention efforts
- **Resource Allocation**: Misclassification affects marketing budget efficiency
- **Business Impact**: Over-reliance on model predictions without human oversight

## Model Monitoring and Maintenance

### Performance Monitoring
- **Drift Detection**: Monthly monitoring of feature distributions
- **Performance Tracking**: Quarterly evaluation of key metrics
- **Business KPIs**: Continuous tracking of churn rates and retention success

### Update Schedule
- **Regular Retraining**: Quarterly with new data
- **Model Validation**: Annual comprehensive model review
- **Emergency Updates**: As needed for significant performance degradation

### Alert Thresholds
- **Accuracy Drop**: Alert if accuracy falls below 75%
- **Data Drift**: Alert if feature distributions change significantly
- **Prediction Drift**: Alert if churn prediction rates deviate >10% from baseline

## Usage Guidelines

### Implementation Recommendations
1. **Batch Predictions**: Recommended for weekly customer risk scoring
2. **Real-time Scoring**: Supported but consider latency requirements
3. **Threshold Tuning**: Adjust based on business cost-benefit analysis
4. **Human Review**: High-value customers should have manual review

### Integration Best Practices
- **API Integration**: Use provided REST API for consistent predictions
- **Data Pipeline**: Ensure feature preprocessing matches training pipeline
- **Monitoring**: Implement prediction logging for model performance tracking
- **Fallback Strategy**: Have backup rules for system failures

## Technical Requirements

### System Requirements
- **Python**: 3.8+
- **TensorFlow**: 2.15.0
- **Memory**: Minimum 2GB RAM
- **CPU**: Standard x86_64 processor
- **Storage**: 100MB for model files

### Dependencies
```
tensorflow==2.15.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
```

## Contact Information

### Model Owner
- **Team**: Data Science Team
- **Contact**: [your.email@company.com]
- **Repository**: [GitHub Repository URL]

### Support
- **Documentation**: See `/docs` folder for detailed guides
- **Issues**: Report via GitHub Issues
- **Updates**: Subscribe to model release notifications

---

*This model card follows the recommended practices for ML model documentation and transparency.*
