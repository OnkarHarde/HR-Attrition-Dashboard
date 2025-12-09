# app.py
# Main Streamlit application for HR Analytics: Employee Attrition Prediction & Insights Dashboard
# Updated: Safe column handling, SHAP safety, and improved training/prediction flow

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

# Validate columns / mapping suggestions
if not df.empty:
    df, mapping_suggestions = validate_and_map_columns(df)
    if mapping_suggestions:
        st.sidebar.info("Column mapping suggestions applied for compatibility with IBM dataset columns.")

# --- Top KPIs ---
st.title("HR Analytics — Employee Attrition Prediction & Insights")
if df.empty:
    st.warning("No dataset available. Upload CSV or use the sample dataset from the sidebar.")
    st.stop()

# Compute engagement score
df["EngagementScore"] = compute_engagement_score(df)

# Filtering
dept_filter = st.sidebar.multiselect("Filter by Department", options=df["Department"].unique().tolist(), default=None)
loc_filter = st.sidebar.multiselect("Filter by BusinessTravel", options=df["BusinessTravel"].unique().tolist(), default=None)

df_filtered = df.copy()
if dept_filter:
    df_filtered = df_filtered[df_filtered["Department"].isin(dept_filter)]
if loc_filter:
    df_filtered = df_filtered[df_filtered["BusinessTravel"].isin(loc_filter)]

# KPI panel
col1, col2, col3, col4 = st.columns(4)
attrition_rate = (df_filtered["Attrition"] == "Yes").mean() if "Attrition" in df_filtered else np.nan
avg_engagement = df_filtered["EngagementScore"].mean()
col1.metric("Attrition Rate", f"{attrition_rate:.2%}")
col2.metric("Average Engagement Score", f"{avg_engagement:.2f}")
col3.metric("Dataset Rows", f"{len(df_filtered)}")
col4.metric("Avg Monthly Income", f"{df_filtered['MonthlyIncome'].mean():.0f}")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Summary", "EDA", "Survival Analysis", "Modeling & Eval", "Explain & Predict"])

# --- Data Summary ---
with tab1:
    st.header("Dataset Preview & Missing Values")
    st.dataframe(df_filtered.head(50))
    st.subheader("Missing Values Report")
    missing = df_filtered.isna().sum().sort_values(ascending=False)
    st.table(missing[missing > 0])

    st.subheader("Distributions for Key Features")
    cols = ["Age", "MonthlyIncome", "JobSatisfaction", "PerformanceRating", "YearsAtCompany", "JobRole"]
    for c in cols:
        if c in df_filtered.columns:
            st.plotly_chart(px.histogram(df_filtered, x=c, nbins=30, title=f"Distribution: {c}"))

# --- EDA ---
with tab2:
    st.header("Exploratory Data Analysis")
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Correlation Heatmap (numeric features)")
        corr = df_filtered[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
        st.plotly_chart(fig)

    st.subheader("Cohort Analysis")
    cohort = cohort_attrition_table(df_filtered)
    st.dataframe(cohort)

    st.subheader("Interactive Charts")
    if "JobRole" in df_filtered.columns and "MonthlyIncome" in df_filtered.columns:
        fig = px.box(df_filtered, x="JobRole", y="MonthlyIncome", points="outliers", title="Income by Job Role")
        st.plotly_chart(fig)

# --- Survival Analysis ---
with tab3:
    st.header("Survival Analysis (Kaplan–Meier)")
    try:
        fig_km = survival_kaplan_meier_plot(df_filtered, time_col="YearsAtCompany", event_col="Attrition", group_col="Department")
        st.plotly_chart(fig_km)
    except Exception as e:
        st.error(f"Survival analysis failed: {e}")

# --- Modeling & Evaluation ---
with tab4:
    st.header("Model Training & Evaluation")
    st.markdown("Configure model, preprocessing, and train. Results will show metrics and plots.")
    random_seed = st.number_input("Random seed", value=42)
    test_size = st.slider("Test size (proportion)", min_value=0.05, max_value=0.4, value=0.2, step=0.05)
    cv_folds = st.slider("CV folds", min_value=2, max_value=10, value=5)

    if st.button("Train model"):
        with st.spinner("Training..."):
            # Filter columns present in dataset
            features_numeric = [c for c in DEFAULT_NUMERIC if c in df.columns]
            features_categorical = [c for c in DEFAULT_CATEGORICAL if c in df.columns]

            if not features_numeric and not features_categorical:
                st.error("No compatible columns found for training. Check dataset columns.")
            else:
                pipeline, model, X_test, y_test = train_model(
                    df,
                    model_choice=model_choice,
                    handle_imbalance=handle_imbalance,
                    random_state=int(random_seed),
                    test_size=float(test_size),
                    cv=cv_folds,
                )
                st.success("Training completed.")

                metrics = evaluate_model(model, pipeline, X_test, y_test)
                st.subheader("Evaluation Metrics")
                st.write(metrics.get("classification_report_dict", {}))
                st.subheader("ROC AUC")
                st.write(metrics.get("roc_auc", "N/A"))
                st.subheader("Confusion Matrix")
                if "confusion_matrix_fig" in metrics:
                    st.plotly_chart(metrics["confusion_matrix_fig"])
                st.subheader("Feature Importance / Coefficients")
                if "feature_importance_fig" in metrics:
                    st.plotly_chart(metrics["feature_importance_fig"])

                st.session_state["pipeline"] = pipeline
                st.session_state["model"] = model
                st.session_state["last_metrics"] = metrics

    # Save / load model
    if save_model_btn and "pipeline" in st.session_state:
        filename = save_model(st.session_state["pipeline"], st.session_state["model"], MODEL_DIR / f"{model_choice.replace(' ','_')}.pkl")
        st.success(f"Model saved to {filename}")

    if load_model_file:
        loaded = load_model(load_model_file)
        st.session_state["pipeline"] = loaded["pipeline"]
        st.session_state["model"] = loaded["model"]
        st.success("Model loaded into session.")

# --- Explain & Predict ---
with tab5:
    st.header("Explainability & Prediction")
    if "pipeline" not in st.session_state or "model" not in st.session_state:
        st.info("No trained model in session. Load a model or train one in Modeling & Eval tab.")
    else:
        pipeline = st.session_state["pipeline"]
        model = st.session_state["model"]
        explainer = SHAPExplainerWrapper(pipeline, model)

        st.subheader("Global SHAP Summary")
        try:
            fig_shap = explainer.summary_plot(show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.error(f"SHAP summary plot failed: {e}")

        st.subheader("Predict an Individual Employee")
        with st.form("individual_predict"):
            Age = st.number_input("Age", value=35)
            MonthlyIncome = st.number_input("MonthlyIncome", value=5000)
            JobSatisfaction = st.selectbox("JobSatisfaction", options=[1, 2, 3, 4], index=2)
            PerformanceRating = st.selectbox("PerformanceRating", options=[1, 2, 3, 4], index=2)
            YearsAtCompany = st.number_input("YearsAtCompany", value=5)
            JobRole = st.selectbox("JobRole", options=df["JobRole"].unique().tolist())
            Department = st.selectbox("Department", options=df["Department"].unique().tolist())
            submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame([{
                "Age": Age,
                "MonthlyIncome": MonthlyIncome,
                "JobSatisfaction": JobSatisfaction,
                "PerformanceRating": PerformanceRating,
                "YearsAtCompany": YearsAtCompany,
                "JobRole": JobRole,
                "Department": Department
            }])
            input_df["EngagementScore"] = compute_engagement_score(input_df)
            try:
                X_proc = preprocess_input_df(input_df, pipeline)
                proba = model.predict_proba(X_proc)[:, 1][0]
                pred_class = model.predict(X_proc)[0]
                st.write(f"Predicted attrition probability: {proba:.3f}")
                st.write(f"Predicted class: {'Yes' if pred_class == 1 else 'No'}")
                try:
                    local_fig = explainer.force_plot(X_proc.iloc[0])
                    st.pyplot(local_fig)
                except Exception as e:
                    st.error(f"SHAP force plot failed: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Dataset-wide prediction export
        if st.button("Predict Attrition for Filtered Dataset"):
            try:
                X = preprocess_input_df(df_filtered, pipeline)
                preds = model.predict_proba(X)[:, 1]
                df_out = df_filtered.copy()
                df_out["Attrition_Prob"] = preds
                df_out["Attrition_Pred"] = (preds >= 0.5).astype(int)
                csv = df_out.to_csv(index=False).encode()
                st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
                high_risk_pct = (df_out["Attrition_Prob"] >= 0.5).mean()
                st.metric("High Risk Employees (%)", f"{high_risk_pct:.2%}")
            except Exception as e:
                st.error(f"Dataset-wide prediction failed: {e}")

# --- SQL snippet ---
st.markdown("---")
st.subheader("SQL Integration Example")
st.code(sql_insert_snippet(), language="sql")
