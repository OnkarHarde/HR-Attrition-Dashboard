# modeling.py
# Data preprocessing, feature engineering, training functions, model evaluation, and model serialization
# Compatible with scikit-learn >=1.2

import pandas as pd
import numpy as np
from typing import Tuple, Any, Dict

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from imblearn.over_sampling import SMOTE
import joblib

import plotly.express as px
import plotly.graph_objects as go


# -----------------
# Defaults
# -----------------

DEFAULT_NUMERIC = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "NumCompaniesWorked",
    "TrainingTimesLastYear",
    "EngagementScore",
]

DEFAULT_CATEGORICAL = [
    "JobRole",
    "Department",
    "EducationField",
    "BusinessTravel",
    "MaritalStatus",
    "Gender",
]


# -----------------
# Preprocessing & Pipeline
# -----------------

def build_column_transformer(
    numeric_cols=DEFAULT_NUMERIC,
    categorical_cols=DEFAULT_CATEGORICAL,
) -> ColumnTransformer:
    """Create ColumnTransformer with numeric and categorical pipelines."""

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor



def build_pipeline_and_model(
    model_choice="Random Forest",
    numeric_cols=None,
    categorical_cols=None,
    random_state=42,
    class_weight=None,
) -> Tuple[Pipeline, Any]:

    """Return preprocessing pipeline and initialized model."""

    if numeric_cols is None:
        numeric_cols = DEFAULT_NUMERIC

    if categorical_cols is None:
        categorical_cols = DEFAULT_CATEGORICAL

    preprocessor = build_column_transformer(numeric_cols, categorical_cols)

    if model_choice == "Logistic Regression":
        model = LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline, model



# -----------------
# Training
# -----------------

def train_model(
    df: pd.DataFrame,
    model_choice="Random Forest",
    handle_imbalance="None",
    random_state=42,
    test_size=0.2,
):
    """Train model pipeline and return pipeline, fitted model, X_test, y_test."""

    df = df.copy()

    if "Attrition" not in df.columns:
        raise ValueError("Attrition column not found in dataset.")

    df = df.dropna(subset=["Attrition"])

    y = (df["Attrition"].astype(str).str.lower() == "yes").astype(int)

    features_numeric = [c for c in DEFAULT_NUMERIC if c in df.columns]
    features_categorical = [c for c in DEFAULT_CATEGORICAL if c in df.columns]

    features = features_numeric + features_categorical
    X = df[features].copy()

    class_weight = "balanced" if handle_imbalance == "class_weight" else None

    pipeline, model = build_pipeline_and_model(
        model_choice=model_choice,
        numeric_cols=features_numeric,
        categorical_cols=features_categorical,
        random_state=random_state,
        class_weight=class_weight,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Handle imbalance
    if handle_imbalance == "SMOTE":
        X_train_proc = pipeline.named_steps["preprocessor"].fit_transform(X_train)
        sm = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sm.fit_resample(X_train_proc, y_train)
        pipeline.named_steps["model"].fit(X_resampled, y_resampled)
    else:
        pipeline.fit(X_train, y_train)

    return pipeline, pipeline.named_steps["model"], X_test, y_test



# -----------------
# Preprocessing new input
# -----------------

def preprocess_input_df(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Apply preprocessing steps from pipeline to new data."""

    return pd.DataFrame(
        pipeline.named_steps["preprocessor"].transform(df),
        columns=pipeline.named_steps["preprocessor"].get_feature_names_out(),
    )



# -----------------
# Evaluation
# -----------------

def evaluate_model(model, pipeline, X_test, y_test) -> Dict[str, Any]:
    """Compute metrics, confusion matrix, and feature importance."""

    X_proc = pd.DataFrame(
        pipeline.named_steps["preprocessor"].transform(X_test),
        columns=pipeline.named_steps["preprocessor"].get_feature_names_out(),
    )

    y_pred = model.predict(X_proc)
    y_proba = model.predict_proba(X_proc)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report_dict": classification_report(
            y_test, y_pred, output_dict=True
        ),
    }

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred No", "Pred Yes"],
            y=["True No", "True Yes"],
            colorscale="Blues",
            showscale=True,
        )
    )
    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    metrics["confusion_matrix_fig"] = fig_cm

    # Feature Importance
    fi_fig = None

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": X_proc.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        fi_fig = px.bar(fi.head(20), x="feature", y="importance", title="Top Feature Importances")

    elif hasattr(model, "coef_"):
        fi = pd.DataFrame(
            {"feature": X_proc.columns, "coef": model.coef_[0]}
        ).sort_values("coef", key=abs, ascending=False)

        fi_fig = px.bar(fi.head(20), x="feature", y="coef", title="Top Coefficients")

    metrics["feature_importance_fig"] = fi_fig

    return metrics



# -----------------
# Model persistence
# -----------------

def save_model(pipeline: Pipeline, model: Any, filepath: str):
    """Save pipeline + model as a single pickle file."""
    joblib.dump({"pipeline": pipeline, "model": model}, filepath)
    return filepath


def load_model(filepath: str):
    """Load saved pipeline + model."""
    return joblib.load(filepath)
