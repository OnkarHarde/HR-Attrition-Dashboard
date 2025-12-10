# modeling.py — Data preprocessing, feature engineering, model training, and serialization

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------
# Helper: Engagement Score
# ---------------------------------------------------------

def compute_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes employee engagement score using:
    JobSatisfaction, WorkLifeBalance, MonthlyIncome percentile, TrainingTimesLastYear
    """
    df = df.copy()
    df["IncomePercentile"] = df["MonthlyIncome"].rank(pct=True)

    df["EngagementScore"] = (
        0.35 * df["JobSatisfaction"] +
        0.25 * df["WorkLifeBalance"] +
        0.25 * df["IncomePercentile"] +
        0.15 * df["TrainingTimesLastYear"]
    )

    return df


# ---------------------------------------------------------
# Preprocessing Builder
# ---------------------------------------------------------

def build_preprocessor(df: pd.DataFrame):
    """
    Creates preprocessing pipeline for numeric and categorical columns.
    Returns the ColumnTransformer.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove target variable
    if "Attrition" in categorical_cols:
        categorical_cols.remove("Attrition")

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


# ---------------------------------------------------------
# Model Trainer
# ---------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    model_name="Logistic Regression",
    test_size=0.2,
    random_state=42,
    use_smote=False,
    return_train_df=False,
):
    """
    Trains an ML model using a preprocessing + model pipeline.
    Supports Logistic Regression and Random Forest.
    """

    df = df.copy()

    # Feature engineering
    df = compute_engagement_score(df)

    # Split features / target
    y = df["Attrition"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Attrition"])

    # Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    # Select model
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced"
        )

    # Train-test split (IMPORTANT: fixed version)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Capture original train indices for SHAP background data
    train_idx = X_train.index

    # SMOTE oversampling (OPTIONAL)
    if use_smote:
        sm = SMOTE(random_state=random_state)
        X_train_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    else:
        X_train_resampled, y_resampled = X_train, y_train

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Fit model
    pipeline.fit(X_train_resampled, y_resampled)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Scores
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    # Cross-validation
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    metrics["cv_roc_auc"] = cv_score.mean()

    # Return pipeline, model, test set, and optional train_df
    if return_train_df:
        train_df = df.loc[train_idx]
        return pipeline, model, X_test, y_test, train_df, metrics

    return pipeline, model, X_test, y_test, metrics


# ---------------------------------------------------------
# Save and Load Models
# ---------------------------------------------------------

def save_model(pipeline, filepath="trained_model.pkl"):
    """Saves trained pipeline to disk."""
    joblib.dump(pipeline, filepath)


def load_model(filepath="trained_model.pkl"):
    """Loads trained pipeline from disk."""
    return joblib.load(filepath)
