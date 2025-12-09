# utils.py
# Helper functions: load data, compute engagement score, survival analysis, cohort analysis, mapping suggestions, sample data generation, SQL snippet

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import plotly.graph_objects as go
import textwrap

def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute Employee Engagement Score as a combination:
    Engagement = 0.4 * normalized(JobSatisfaction) + 0.2 * normalized(WorkLifeBalance)
                 + 0.3 * income_percentile + 0.1 * normalized(TrainingTimesLastYear)
    Expect missing columns to be handled gracefully.
    """
    s = pd.Series(0, index=df.index, dtype=float)
    # JobSatisfaction 1-4
    if "JobSatisfaction" in df.columns:
        js = (df["JobSatisfaction"].fillna(df["JobSatisfaction"].median()) - 1) / 3.0
    else:
        js = 0
    if "WorkLifeBalance" in df.columns:
        wlb = (df["WorkLifeBalance"].fillna(df["WorkLifeBalance"].median()) - 1) / 3.0
    else:
        wlb = 0
    if "MonthlyIncome" in df.columns:
        inc_pct = df["MonthlyIncome"].rank(pct=True).fillna(0)
    else:
        inc_pct = 0
    if "TrainingTimesLastYear" in df.columns:
        train_norm = df["TrainingTimesLastYear"].fillna(0) / (df["TrainingTimesLastYear"].max() if df["TrainingTimesLastYear"].max() > 0 else 1)
    else:
        train_norm = 0

    s = 0.4 * js + 0.2 * wlb + 0.3 * inc_pct + 0.1 * train_norm
    # scale to 0-100
    s = (s - s.min()) / (s.max() - s.min() + 1e-9) * 100
    return s

def sample_synthetic_rows(n=100) -> pd.DataFrame:
    """Generate synthetic rows resembling IBM Attrition dataset columns for quick testing."""
    rng = np.random.default_rng(42)
    job_roles = ["Research Scientist", "Sales Executive", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Human Resources"]
    departments = ["Research & Development", "Sales", "Human Resources"]
    df = pd.DataFrame({
        "Age": rng.integers(18, 60, n),
        "MonthlyIncome": rng.integers(2000, 20000, n),
        "JobSatisfaction": rng.integers(1, 5, n),
        "PerformanceRating": rng.integers(1, 5, n),
        "YearsAtCompany": rng.integers(0, 40, n),
        "JobRole": rng.choice(job_roles, n),
        "Department": rng.choice(departments, n),
        "BusinessTravel": rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n),
        "NumCompaniesWorked": rng.integers(0, 10, n),
        "TrainingTimesLastYear": rng.integers(0, 10, n),
        "Attrition": rng.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "WorkLifeBalance": rng.integers(1, 5, n),
        "EducationField": rng.choice(["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"], n),
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
        "Gender": rng.choice(["Male", "Female"], n),
    })
    df["EngagementScore"] = compute_engagement_score(df)
    return df

def validate_and_map_columns(df: pd.DataFrame) -> (pd.DataFrame, dict):
    """
    Check for expected IBM column names and attempt to map close matches.
    Returns possibly-updated df and mapping suggestions applied.
    """
    expected = {"Age","JobSatisfaction","MonthlyIncome","YearsAtCompany","JobRole","Department","Attrition","TrainingTimesLastYear","WorkLifeBalance"}
    existing = set(df.columns)
    mapping = {}
    # simple lowercase matching
    for e in expected:
        if e not in existing:
            lower_map = {c.lower(): c for c in existing}
            if e.lower() in lower_map:
                mapping[e] = lower_map[e.lower()]
    if mapping:
        df = df.rename(columns={v:k for k,v in mapping.items()})
    return df, mapping

def survival_kaplan_meier_plot(df: pd.DataFrame, time_col="YearsAtCompany", event_col="Attrition", group_col=None):
    """
    Returns a plotly figure with Kaplan-Meier curves, grouped by group_col if provided.
    Event_col expected to be "Yes"/"No" strings or booleans.
    """
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        for g in groups:
            sub = df[df[group_col] == g]
            T = sub[time_col].fillna(0)
            E = (sub[event_col].astype(str).str.lower() == "yes").astype(int)
            if len(T) == 0:
                continue
            kmf.fit(T, event_observed=E, label=str(g))
            fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[kmf.survival_function_.columns[0]], mode="lines", name=str(g)))
    else:
        T = df[time_col].fillna(0)
        E = (df[event_col].astype(str).str.lower() == "yes").astype(int)
        kmf.fit(T, event_observed=E, label="All")
        fig.add_trace(go.Scatter(x=kmf.survival_function_.index, y=kmf.survival_function_[kmf.survival_function_.columns[0]], mode="lines", name="All"))
    fig.update_layout(title="Kaplan-Meier Survival Curve", xaxis_title=time_col, yaxis_title="Survival Probability")
    return fig

def cohort_attrition_table(df: pd.DataFrame):
    """
    Return a small DataFrame with attrition rates by Department and JobRole.
    """
    if "Attrition" not in df.columns:
        return pd.DataFrame()
    dfc = df.copy()
    dfc["AttritionFlag"] = (dfc["Attrition"].astype(str).str.lower() == "yes").astype(int)
    table = dfc.groupby(["Department","JobRole"]).agg(
        total=("AttritionFlag","size"),
        attrition_count=("AttritionFlag","sum"),
    ).reset_index()
    table["attrition_rate"] = table["attrition_count"] / table["total"]
    return table.sort_values("attrition_rate", ascending=False)

def sql_insert_snippet():
    """
    Return a SQL snippet (string) showing parameterized insert for predictions into a table.
    """
    snippet = textwrap.dedent("""
    -- Example parameterized insert (Python + pymysql or psycopg2)
    INSERT INTO employee_attrition_predictions
    (employee_id, prediction_date, attrition_prob, attrition_pred, model_version)
    VALUES (%s, %s, %s, %s, %s);
    """).strip()
    return snippet
