import pandas as pd

CATEGORICAL = [
    'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'
]
NUMERICAL = ['tenure','MonthlyCharges','TotalCharges']

TARGET = 'Churn'

def load_data(path='data/telco.csv'):
    df = pd.read_csv(path)
    # Clean TotalCharges to numeric (Telco quirk)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', None), errors='coerce')
    # Basic drop of rows with missing critical fields
    df = df.dropna(subset=['tenure','MonthlyCharges','TotalCharges'])
    # Standardize target
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({'Yes':1,'No':0}).astype(int)
    return df
