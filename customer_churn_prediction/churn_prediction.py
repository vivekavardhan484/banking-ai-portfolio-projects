"""
Customer Churn Prediction
Author: Vivek Kothapalli

This project demonstrates how machine learning can predict customers who are
likely to leave a bank or financial service provider.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def create_synthetic_churn_data(n_samples=8000, random_state=42):
    """
    Creates a synthetic customer churn dataset.
    """
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 75, size=n_samples)
    tenure_months = rng.integers(1, 120, size=n_samples)
    monthly_charges = rng.normal(65, 20, size=n_samples).clip(10, 160)
    num_products = rng.integers(1, 5, size=n_samples)
    credit_score = rng.normal(650, 90, size=n_samples).clip(300, 850)
    support_calls = rng.poisson(1.2, size=n_samples)
    complaints = rng.binomial(1, 0.18, size=n_samples)
    online_banking_active = rng.binomial(1, 0.70, size=n_samples)
    missed_payments = rng.poisson(0.3, size=n_samples)

    churn_risk = (
        0.04 * monthly_charges -
        0.025 * tenure_months +
        0.9 * complaints +
        0.5 * support_calls +
        0.7 * missed_payments -
        0.8 * online_banking_active -
        0.004 * credit_score
    )

    probability = 1 / (1 + np.exp(-(churn_risk + 0.5)))
    churn = rng.binomial(1, probability)

    df = pd.DataFrame({
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "num_products": num_products,
        "credit_score": credit_score.round(0).astype(int),
        "support_calls": support_calls,
        "complaints": complaints,
        "online_banking_active": online_banking_active,
        "missed_payments": missed_payments,
        "churn": churn
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


def save_feature_importance(model, feature_names):
    """
    Saves a feature importance chart for tree-based models.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Feature Importance - Customer Churn")
    plt.tight_layout()
    plt.savefig("churn_feature_importance.png", bbox_inches="tight")
    plt.close()


def main():
    df = create_synthetic_churn_data()

    print("Dataset preview:")
    print(df.head())

    print("\nChurn class distribution:")
    print(df["churn"].value_counts(normalize=True).rename("proportion"))

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    random_forest_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
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

        if name == "Random Forest":
            best_score = score
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} with ROC-AUC {best_score:.3f}")

    joblib.dump(best_model, "churn_prediction_model.pkl")

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {best_name}")
    plt.savefig("churn_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    if best_name == "Random Forest":
        save_feature_importance(best_model, X.columns)

    print("\nSaved files:")
    print("- churn_prediction_model.pkl")
    print("- churn_confusion_matrix.png")
    if best_name == "Random Forest":
        print("- churn_feature_importance.png")


if __name__ == "__main__":
    main()
