# 📊 Data Dictionary

## Dataset Overview

The Telco Customer Churn dataset contains information about a telecommunications company's customers and whether they churned (left the company) or not. This dataset is commonly used for binary classification problems in machine learning.

## Target Variable

| Variable | Type | Description | Values |
|----------|------|-------------|---------|
| **Churn** | Categorical | Whether the customer churned | `Yes`, `No` |

## Feature Variables

### 📋 Customer Demographics

| Variable | Type | Description | Values | Notes |
|----------|------|-------------|---------|-------|
| **customerID** | String | Unique customer identifier | Alphanumeric | Removed during preprocessing |
| **gender** | Categorical | Customer's gender | `Male`, `Female` | Encoded as 0/1 |
| **SeniorCitizen** | Binary | Whether customer is senior citizen | `0`, `1` | 0=No, 1=Yes |
| **Partner** | Categorical | Whether customer has a partner | `Yes`, `No` | Encoded as 1/0 |
| **Dependents** | Categorical | Whether customer has dependents | `Yes`, `No` | Encoded as 1/0 |

### 📞 Phone Services

| Variable | Type | Description | Values | Notes |
|----------|------|-------------|---------|-------|
| **PhoneService** | Categorical | Whether customer has phone service | `Yes`, `No` | Encoded as 1/0 |
| **MultipleLines** | Categorical | Whether customer has multiple lines | `Yes`, `No`, `No phone service` | Standardized to Yes/No |

### 🌐 Internet Services

| Variable | Type | Description | Values | Notes |
|----------|------|-------------|---------|-------|
| **InternetService** | Categorical | Type of internet service | `DSL`, `Fiber optic`, `No` | One-hot encoded |
| **OnlineSecurity** | Categorical | Whether customer has online security | `Yes`, `No`, `No internet service` | Standardized to Yes/No |
| **OnlineBackup** | Categorical | Whether customer has online backup | `Yes`, `No`, `No internet service` | Standardized to Yes/No |
| **DeviceProtection** | Categorical | Whether customer has device protection | `Yes`, `No`, `No internet service` | Standardized to Yes/No |
| **TechSupport** | Categorical | Whether customer has tech support | `Yes`, `No`, `No internet service` | Standardized to Yes/No |
| **StreamingTV** | Categorical | Whether customer has streaming TV | `Yes`, `No`, `No internet service` | Standardized to Yes/No |
| **StreamingMovies** | Categorical | Whether customer has streaming movies | `Yes`, `No`, `No internet service` | Standardized to Yes/No |

### 💳 Account Information

| Variable | Type | Description | Values | Notes |
|----------|------|-------------|---------|-------|
| **tenure** | Numerical | Number of months customer has stayed | 0-72 | Continuous variable, scaled |
| **Contract** | Categorical | Contract term | `Month-to-month`, `One year`, `Two year` | One-hot encoded |
| **PaperlessBilling** | Categorical | Whether customer uses paperless billing | `Yes`, `No` | Encoded as 1/0 |
| **PaymentMethod** | Categorical | Payment method | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | One-hot encoded |
| **MonthlyCharges** | Numerical | Monthly charge amount | 18.25-118.75 | Continuous variable, scaled |
| **TotalCharges** | Numerical | Total amount charged | 18.8-8684.8 | Continuous variable, scaled |

## Data Preprocessing Steps

### 1. Data Cleaning
- **Remove customerID**: Not useful for prediction
- **Handle TotalCharges**: Convert from string to numeric, remove empty values
- **Remove duplicates**: Ensure data integrity
- **Standardize categories**: Convert "No internet service" and "No phone service" to "No"

### 2. Feature Engineering
- **Binary encoding**: Convert Yes/No categorical variables to 1/0
- **One-hot encoding**: Create dummy variables for multi-category features
- **Feature scaling**: Apply MinMax scaling to numerical features (tenure, MonthlyCharges, TotalCharges)

### 3. Final Dataset Structure
- **Total features**: 26 (after encoding)
- **Numerical features**: 3 (tenure, MonthlyCharges, TotalCharges)
- **Binary features**: 12 (gender, Partner, Dependents, etc.)
- **One-hot encoded features**: 11 (InternetService, Contract, PaymentMethod)

## Feature Importance Insights

Based on model analysis, the most important features for churn prediction are:

1. **Contract_Month-to-month** (15.2%) - Month-to-month contracts show highest churn
2. **tenure** (12.1%) - Longer tenure customers are less likely to churn
3. **TotalCharges** (10.3%) - Higher total charges may indicate satisfaction
4. **MonthlyCharges** (8.7%) - Very high monthly charges may lead to churn
5. **InternetService_Fiber optic** (7.4%) - Fiber optic users show higher churn rates
6. **PaymentMethod_Electronic check** (6.2%) - Electronic check users more likely to churn

## Data Quality Notes

- **Missing values**: 11 rows with empty TotalCharges (removed during preprocessing)
- **Data types**: All features converted to numerical for model training
- **Outliers**: No significant outliers detected in numerical features
- **Class balance**: Approximately 73% No Churn, 27% Churn (slightly imbalanced)

## Business Context

Understanding these features helps in:
- **Customer retention strategies**: Target month-to-month contract customers
- **Service improvements**: Address fiber optic service issues
- **Payment optimization**: Encourage automatic payment methods
- **Risk assessment**: Monitor new customers (low tenure) closely

## Usage in Model

All features are used in the neural network model after preprocessing:
- Input layer: 26 neurons (one for each feature)
- Features are normalized/standardized for optimal model performance
- Target variable (Churn) is binary encoded (1=Yes, 0=No)
