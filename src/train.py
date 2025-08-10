import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

from utils import load_data, CATEGORICAL, NUMERICAL, TARGET

def get_preprocessor():
    numeric_pipe = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ('num', numeric_pipe, NUMERICAL),
            ('cat', cat_pipe, CATEGORICAL)
        ]
    )
    return pre

def evaluate(model, X_test, y_test, out_dir=Path('models')):
    pred = model.predict(X_test)
    proba = getattr(model, 'predict_proba', lambda X: np.column_stack([1-model.predict(X), model.predict(X)]))(X_test)[:,1]
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))
    print("ROC-AUC:", roc_auc_score(y_test, proba))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No','Yes'])
    plt.yticks(tick_marks, ['No','Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(out_dir / 'confusion_matrix.png', bbox_inches='tight')

def main():
    df = load_data('data/telco.csv')
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    preprocessor = get_preprocessor()

    # Class weights to mitigate imbalance
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    cw = {c:w for c,w in zip(classes, weights)}

    # Logistic Regression baseline
    logreg = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight=cw))
    ])

    # Random Forest
    rf = Pipeline(steps=[
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, class_weight='balanced'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # CV scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, pipe in [('LogReg', logreg), ('RandomForest', rf)]:
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"{name} CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit best (choose RF by default; adjust based on scores)
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test)

    # Persist model
    Path('models').mkdir(exist_ok=True, parents=True)
    joblib.dump(rf, 'models/churn_pipeline.joblib')
    print('Saved model to models/churn_pipeline.joblib')

if __name__ == '__main__':
    main()
