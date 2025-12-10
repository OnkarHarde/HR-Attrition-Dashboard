# app.py
# Main Streamlit application for HR Analytics: Employee Attrition Prediction & Insights Dashboard
# Updated: Safe SHAP initialization + passes background data + fixed force plot + stability improvements

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from modeling import (
    build_pipeline_and_model,
    train_model,
    load_model,
    save_model,
    evaluate_model,
    DEFAULT_NUMERIC,
    DEFAULT_CATEGORICAL,
    preprocess_input_df,
)
from explainer import SHAPExplainerWrapper
from utils import (
    compute_engagement_score,
    sample_synthetic_rows,
    validate_and_map_columns,
    survival_kaplan_meier_plot,
    cohort_attrition_table,
    sql_insert_snippet,
)
import plotly.express as px

# --- Setup ---
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar ---
st.sidebar.title("HR Attrition Dashboard")
st.sidebar.markdown("Upload the IBM Attrition Dataset CSV or use the sample dataset.")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample synthetic data instead", value=False)

model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
handle_imbalance = st.sidebar.selectbox("Handle Imbalance", ["None", "class_weight", "SMOTE"])
save_model_btn = st.sidebar.button("Save last trained model")
load_model_file = st.sidebar.file_uploader("Load model (.pkl)", type=["pkl"])
predict_from_saved = st.sidebar.checkbox("Use loaded model for predictions", value=False)

# --- Load dataset ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = sample_synthetic_rows(n=200)
else:
    try:
        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    except FileNotFoundError:
        st.sidebar.warning("No dataset found. Toggle 'Use sample synthetic data' or upload a dataset.")
        df = pd.DataFrame()

# Validate columns
if not df.empty:
    df, mapping_suggestions = validate_and_map_columns(df)
    if mapping_suggestions:
        st.sidebar.info("Column mapping suggestions applied for compatibility.")

# Page Heading
st.title("HR Analytics — Employee Attrition Prediction & Insights")

if df.empty:
    st.warning("No dataset available. Upload or use sample dataset.")
    st.stop()

# Compute engagement score
df["EngagementScore"] = compute_engagement_score(df)

# Filters
dept_filter = st.sidebar.multiselect("Filter by Department", options=df["Department"].unique().tolist(), default=None)
loc_filter = st.sidebar.multiselect("Filter by BusinessTravel", options=df["BusinessTravel"].unique().tolist(), default=None)

df_filtered = df.copy()
if dept_filter:
    df_filtered = df_filtered[df_filtered["Department"].isin(dept_filter)]
if loc_filter:
    df_filtered = df_filtered[df_filtered["BusinessTravel"].isin(loc_filter)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
attrition_rate = (df_filtered["Attrition"] == "Yes").mean()
avg_engagement = df_filtered["EngagementScore"].mean()

col1.metric("Attrition Rate", f"{attrition_rate:.2%}")
col2.metric("Avg Engagement Score", f"{avg_engagement:.2f}")
col3.metric("Dataset Rows", f"{len(df_filtered)}")
col4.metric("Avg Monthly Income", f"{df_filtered['MonthlyIncome'].mean():.0f}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Summary", "EDA", "Survival Analysis", "Modeling & Eval", "Explain & Predict"])

# --- Tab 1 ---
with tab1:
    st.header("Dataset Preview & Missing Values")
    st.dataframe(df_filtered.head(50))

    st.subheader("Missing Values")
    missing = df_filtered.isna().sum()
    st.table(missing[missing > 0])

    st.subheader("Key Feature Distributions")
    cols = ["Age", "MonthlyIncome", "JobSatisfaction", "PerformanceRating", "YearsAtCompany", "JobRole"]
    for c in cols:
        if c in df_filtered.columns:
            st.plotly_chart(px.histogram(df_filtered, x=c, nbins=30, title=f"Distribution: {c}"))

# --- Tab 2 ---
with tab2:
    st.header("Exploratory Data Analysis")

    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        corr = df_filtered[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

    st.subheader("Cohort Attrition")
    st.dataframe(cohort_attrition_table(df_filtered))

    st.subheader("Income by Job Role")
    if "JobRole" in df_filtered and "MonthlyIncome" in df_filtered:
        st.plotly_chart(px.box(df_filtered, x="JobRole", y="MonthlyIncome", points="outliers"))

# --- Tab 3 ---
with tab3:
    st.header("Survival Analysis")
    try:
        fig = survival_kaplan_meier_plot(df_filtered, "YearsAtCompany", "Attrition", "Department")
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Survival analysis failed: {e}")

# --- Tab 4 ---
with tab4:
    st.header("Model Training & Evaluation")

    random_seed = st.number_input("Random Seed", value=42)
    test_size = st.slider("Test Size", 0.05, 0.4, 0.2)
    cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)

    if st.button("Train Model"):
        with st.spinner("Training..."):
            pipeline, model, X_test, y_test, train_df = train_model(
                df,
                model_choice=model_choice,
                handle_imbalance=handle_imbalance,
                random_state=int(random_seed),
                test_size=float(test_size),
                cv=cv_folds,
                return_train_df=True
            )

            st.session_state["pipeline"] = pipeline
            st.session_state["model"] = model
            st.session_state["background_df"] = train_df.sample(min(100, len(train_df)), random_state=42)

            metrics = evaluate_model(model, pipeline, X_test, y_test)
            st.write(metrics.get("classification_report_dict", {}))

            if "confusion_matrix_fig" in metrics:
                st.plotly_chart(metrics["confusion_matrix_fig"])
            if "feature_importance_fig" in metrics:
                st.plotly_chart(metrics["feature_importance_fig"])

    if save_model_btn and "pipeline" in st.session_state:
        filename = save_model(st.session_state["pipeline"], st.session_state["model"], MODEL_DIR / f"{model_choice}.pkl")
        st.success(f"Model saved to: {filename}")

    if load_model_file:
        loaded = load_model(load_model_file)
        st.session_state["pipeline"] = loaded["pipeline"]
        st.session_state["model"] = loaded["model"]
        st.success("Model loaded successfully.")

# --- Tab 5 ---
with tab5:
    st.header("Explainability & Prediction")

    if "pipeline" not in st.session_state:
        st.info("Train or load a model first.")
        st.stop()

    pipeline = st.session_state["pipeline"]
    model = st.session_state["model"]
    background_df = st.session_state.get("background_df", df.sample(50))

    # Initialize explainer safely
    try:
        explainer = SHAPExplainerWrapper(pipeline, model, background_data=background_df)
    except Exception as e:
        st.error(f"SHAP initialization failed: {e}")
        st.stop()

    st.subheader("Global SHAP Summary")
    try:
        fig = explainer.summary_plot(show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP summary plot failed: {e}")

    st.subheader("Predict an Employee")
    with st.form("pred_form"):
        Age = st.number_input("Age", 18, 60, 35)
        MonthlyIncome = st.number_input("MonthlyIncome", 1000, 30000, 5000)
        JobSatisfaction = st.selectbox("JobSatisfaction", [1, 2, 3, 4])
        PerformanceRating = st.selectbox("PerformanceRating", [1, 2, 3, 4])
        YearsAtCompany = st.number_input("YearsAtCompany", 0, 40, 5)
        JobRole = st.selectbox("JobRole", df["JobRole"].unique())
        Department = st.selectbox("Department", df["Department"].unique())
        go = st.form_submit_button("Predict")

    if go:
        row = pd.DataFrame([{
            "Age": Age,
            "MonthlyIncome": MonthlyIncome,
            "JobSatisfaction": JobSatisfaction,
            "PerformanceRating": PerformanceRating,
            "YearsAtCompany": YearsAtCompany,
            "JobRole": JobRole,
            "Department": Department,
        }])

        row["EngagementScore"] = compute_engagement_score(row)

        try:
            Xp = preprocess_input_df(row, pipeline)
            proba = model.predict_proba(Xp)[0, 1]
            pred = model.predict(Xp)[0]

            st.write(f"Attrition Probability: {proba:.3f}")
            st.write(f"Prediction: {'Yes' if pred == 1 else 'No'}")

            # Local SHAP
            fig_local = explainer.force_plot(row)
            st.pyplot(fig_local)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    if st.button("Predict for whole dataset"):
        try:
            X = preprocess_input_df(df_filtered, pipeline)
            preds = model.predict_proba(X)[:, 1]
            df_out = df_filtered.copy()
            df_out["Attrition_Prob"] = preds
            df_out["Attrition_Pred"] = (preds >= 0.5).astype(int)
            st.download_button("Download Predictions CSV", df_out.to_csv(index=False), "predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.subheader("SQL Example")
st.code(sql_insert_snippet(), language="sql")
