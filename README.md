# Customer Churn Prediction — Portfolio Edition

This is a polished, end-to-end churn prediction project suitable for your portfolio.

## Features
- Clean preprocessing with `ColumnTransformer` and `Pipeline`
- Imbalance handling with class weights (optionally SMOTE)
- Benchmark: Logistic Regression, Random Forest, XGBoost (plug-in), with cross-validation
- Rich evaluation: confusion matrix, ROC-AUC, classification report
- Explainability with SHAP (for tree models)
- Reusable artifacts: serialized pipeline via `joblib`
- Streamlit app for interactive predictions

## Structure
```
churn_portfolio/
  ├── data/                      # Put Telco churn CSV here as telco.csv
  ├── models/                    # Saved model artifacts
  ├── notebooks/                 # EDA & storytelling notebooks
  ├── src/
  │   ├── train.py               # Train & evaluate, saves pipeline
  │   └── utils.py               # Helper functions
  ├── app.py                     # Streamlit app
  └── requirements.txt
```

## Quickstart
1. Make a new virtual environment and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your dataset at `data/telco.csv`. (The common one is WA_Fn-UseC_-Telco-Customer-Churn.csv; rename to telco.csv.)
3. Train:
   ```bash
   python src/train.py
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
