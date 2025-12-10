# app.py
# Main Streamlit application for HR Analytics: Employee Attrition Prediction & Insights Dashboard
# Updated with fixed SHAP logic, correct train_model return values, and improved prediction pipeline

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from modeling import (
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
import shap

# --- Setup ---
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# --- Sidebar ---
st.sidebar.title("HR Attrition Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload IBM Attrition Dataset CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample synthetic dataset", value=False)

model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
handle_imbalance = st.sidebar.selectbox("Handle Class Imbalance", ["None", "class_weight", "SMOTE"])

save_model_btn = st.sidebar.button("Save trained model")
load_model_file = st.sidebar.file_uploader("Load Saved Model (.pkl)", type=["pkl"])
predict_from_saved = st.sidebar.checkbox("Use loaded model for predictions", value=False)


# --- Load dataset ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif use_sample:
    df = sample_synthetic_rows(n=200)
else:
    try:
        df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    except:
        st.warning("Upload dataset or enable sample dataset.")
        df = pd.DataFrame()

if not df.empty:
    df, mapping = validate_and_map_columns(df)
    df["EngagementScore"] = compute_engagement_score(df)

# Stop if no data
if df.empty:
    st.stop()

# --- KPI Panel ---
st.title("HR Analytics — Attrition Prediction & Insights")

dept_filter = st.sidebar.multiselect(
    "Filter by Department",
    options=df["Department"].unique().tolist(),
)

df_filtered = df.copy()
if dept_filter:
    df_filtered = df_filtered[df_filtered["Department"].isin(dept_filter)]

col1, col2, col3 = st.columns(3)
col1.metric("Attrition Rate", f"{(df_filtered['Attrition']=='Yes').mean():.2%}")
col2.metric("Avg Engagement Score", f"{df_filtered['EngagementScore'].mean():.2f}")
col3.metric("Rows", f"{len(df_filtered)}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Summary",
    "EDA",
    "Survival Analysis",
    "Modeling",
    "Explain & Predict"
])

# ===============================
# DATA SUMMARY
# ===============================
with tab1:
    st.header("Dataset Preview")
    st.dataframe(df_filtered.head())

    st.subheader("Missing Values")
    st.table(df_filtered.isna().sum()[lambda x: x > 0])

    st.subheader("Distributions")
    cols = ["Age", "MonthlyIncome", "JobSatisfaction", "PerformanceRating", "YearsAtCompany"]
    for c in cols:
        if c in df_filtered:
            st.plotly_chart(px.histogram(df_filtered, x=c, nbins=30))


# ===============================
# EDA
# ===============================
with tab2:
    st.header("Correlation Heatmap")
    num_cols = df_filtered.select_dtypes(include=np.number).columns
    if len(num_cols) > 1:
        fig = px.imshow(df_filtered[num_cols].corr(), text_auto=True)
        st.plotly_chart(fig)

    st.header("Cohort Attrition")
    coh = cohort_attrition_table(df_filtered)
    st.dataframe(coh)

    st.header("Boxplot: Income by Job Role")
    if "JobRole" in df_filtered and "MonthlyIncome" in df_filtered:
        st.plotly_chart(px.box(df_filtered, x="JobRole", y="MonthlyIncome"))


# ===============================
# SURVIVAL ANALYSIS
# ===============================
with tab3:
    st.header("Survival Analysis")
    try:
        km_fig = survival_kaplan_meier_plot(
            df_filtered, "YearsAtCompany", "Attrition", "Department"
        )
        st.plotly_chart(km_fig)
    except Exception as e:
        st.error(f"Survival analysis failed: {e}")


# ===============================
# MODELING
# ===============================
with tab4:
    st.header("Train Model")

    seed = st.number_input("Random Seed", value=42)
    test_size = st.slider("Test Size", 0.05, 0.4, 0.2)

    if st.button("Train Model Now"):
        with st.spinner("Training..."):
            pipeline, model, X_test, y_test, train_df = train_model(
                df,
                model_choice=model_choice,
                handle_imbalance=handle_imbalance,
                random_state=int(seed),
                test_size=float(test_size),
                return_train_df=True
            )

            st.session_state["pipeline"] = pipeline
            st.session_state["model"] = model
            st.session_state["train_df"] = train_df

            st.success("Training Complete")

            metrics = evaluate_model(model, pipeline, X_test, y_test)
            st.subheader("Metrics")
            st.json(metrics["classification_report_dict"])
            st.subheader("Confusion Matrix")
            st.plotly_chart(metrics["confusion_matrix_fig"])
            st.subheader("Feature Importance")
            st.plotly_chart(metrics["feature_importance_fig"])

    # --- Save / Load Model ---
    if save_model_btn and "pipeline" in st.session_state:
        path = save_model(
            st.session_state["pipeline"],
            st.session_state["model"],
            MODEL_DIR / f"{model_choice.replace(' ','_')}.pkl"
        )
        st.success(f"Model saved: {path}")

    if load_model_file:
        loaded = load_model(load_model_file)
        st.session_state["pipeline"] = loaded["pipeline"]
        st.session_state["model"] = loaded["model"]
        st.success("Model loaded successfully.")


# ===============================
# SHAP + PREDICT
# ===============================
with tab5:
    st.header("Explainability & Prediction")

    if "pipeline" not in st.session_state:
        st.warning("Train or load a model first.")
        st.stop()

    pipeline = st.session_state["pipeline"]
    model = st.session_state["model"]
    train_df = st.session_state.get("train_df")

    # Initialize SHAP correctly
    try:
        explainer = SHAPExplainerWrapper(
            pipeline=pipeline,
            model=model,
            background_df=train_df.drop(columns=["Attrition"])
        )
    except Exception as e:
        st.error(f"SHAP initialization failed: {e}")
        st.stop()

    # GLOBAL SUMMARY PLOT
    st.subheader("Global SHAP Summary")
    try:
        shap_values = explainer.explain(train_df.drop(columns=["Attrition"]).head(50))
        fig = shap.summary_plot(shap_values.values, feature_names=explainer.explainer.feature_names, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP summary failed: {e}")

    # INDIVIDUAL PREDICTION
    st.subheader("Predict Individual Employee")

    with st.form("predict_form"):
        input_dict = {}

        for col in DEFAULT_NUMERIC:
            if col in df.columns:
                input_dict[col] = st.number_input(col, value=float(df[col].median()))

        for col in DEFAULT_CATEGORICAL:
            if col in df.columns:
                input_dict[col] = st.selectbox(col, options=df[col].unique().tolist())

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([input_dict])
        input_df["EngagementScore"] = compute_engagement_score(input_df)

        X_proc = preprocess_input_df(input_df, pipeline)
        proba = model.predict_proba(X_proc)[:, 1][0]
        pred = "Yes" if proba >= 0.5 else "No"

        st.success(f"Attrition Probability: {proba:.3f} — Prediction: {pred}")

        # Local SHAP
        try:
            shap_vals = explainer.explain(input_df)
            fig2 = shap.force_plot(shap_vals.values[0], matplotlib=True, show=False)
            st.pyplot(fig2)
        except Exception as e:
            st.error(f"Local SHAP failed: {e}")

    # Export predictions
    st.subheader("Dataset-Wide Predictions")
    if st.button("Predict for Full Dataset"):
        X_all = preprocess_input_df(df_filtered, pipeline)
        preds = model.predict_proba(X_all)[:, 1]

        df_out = df_filtered.copy()
        df_out["Attrition_Prob"] = preds
        df_out["Prediction"] = (preds >= 0.5).astype(int)

        csv = df_out.to_csv(index=False).encode()
        st.download_button("Download Predictions CSV", csv, "predictions.csv")

# SQL snippet
st.markdown("---")
st.subheader("SQL Integration")
st.code(sql_insert_snippet(), language="sql")
