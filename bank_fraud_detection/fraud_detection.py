"""
Bank Fraud Detection System
Author: Vivek Kothapalli

This project demonstrates a complete machine learning workflow for detecting
potentially fraudulent banking transactions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def create_synthetic_fraud_data(n_samples=10000, fraud_rate=0.03, random_state=42):
    """
    Creates a synthetic fraud dataset with realistic transaction-related features.
    """
    rng = np.random.default_rng(random_state)

    amount = rng.exponential(scale=80, size=n_samples)
    transaction_hour = rng.integers(0, 24, size=n_samples)
    account_age_days = rng.integers(1, 3650, size=n_samples)
    num_prev_transactions = rng.poisson(lam=25, size=n_samples)
    is_international = rng.binomial(1, 0.15, size=n_samples)
    device_change = rng.binomial(1, 0.10, size=n_samples)
    failed_login_attempts = rng.poisson(lam=0.4, size=n_samples)

    # Create fraud probability based on risk patterns
    risk_score = (
        0.015 * amount +
        1.2 * is_international +
        1.8 * device_change +
        0.9 * failed_login_attempts +
        0.8 * ((transaction_hour >= 0) & (transaction_hour <= 5)).astype(int) -
        0.0002 * account_age_days
    )

    probability = 1 / (1 + np.exp(-(risk_score - np.percentile(risk_score, 100 * (1 - fraud_rate)))))
    is_fraud = rng.binomial(1, probability)

    df = pd.DataFrame({
        "amount": amount.round(2),
        "transaction_hour": transaction_hour,
        "account_age_days": account_age_days,
        "num_prev_transactions": num_prev_transactions,
        "is_international": is_international,
        "device_change": device_change,
        "failed_login_attempts": failed_login_attempts,
        "is_fraud": is_fraud
    })

    return df


def evaluate_model(model, X_test, y_test, model_name):
    """
    Prints classification metrics and returns ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")

    return roc_auc_score(y_test, y_proba)


def main():
    df = create_synthetic_fraud_data()

    print("Dataset preview:")
    print(df.head())

    print("\nFraud class distribution:")
    print(df["is_fraud"].value_counts(normalize=True).rename("proportion"))

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": random_forest_model
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        score = evaluate_model(model, X_test, y_test, name)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} with ROC-AUC {best_score:.3f}")

    # Save best model
    joblib.dump(best_model, "fraud_detection_model.pkl")

    # Save confusion matrix
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("fraud_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Save ROC curve
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title(f"ROC Curve - {best_name}")
    plt.savefig("fraud_roc_curve.png", bbox_inches="tight")
    plt.close()

    print("\nSaved files:")
    print("- fraud_detection_model.pkl")
    print("- fraud_confusion_matrix.png")
    print("- fraud_roc_curve.png")


if __name__ == "__main__":
    main()
