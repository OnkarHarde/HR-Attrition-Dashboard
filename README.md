# HR Analytics: Employee Attrition Prediction & Insights Dashboard

## Problem Statement
Employee attrition is a critical challenge for organizations, leading to increased recruitment costs, loss of talent, and reduced productivity. This project aims to predict which employees are at risk of leaving and provide actionable insights for HR teams using historical employee data.

---

## What We Did
- Built a **Streamlit dashboard** to explore, analyze, and predict employee attrition.
- Performed **EDA** (data summary, distributions, correlations, cohort analysis).
- Created **Employee Engagement Score** as a key metric.
- Conducted **Survival Analysis** using Kaplan-Meier curves.
- Trained **Logistic Regression** and **Random Forest** models with imbalance handling.
- Integrated **SHAP explainability** for feature impact.
- Enabled **predictions** for individual employees and bulk dataset.
- Provided an **example SQL integration** to store predictions.

---

## What We Predict
- **Target:** `Attrition` (Yes/No)
- **Outputs:**
  - Attrition probability for each employee
  - Predicted class (Yes/No)
  - Key factors contributing to prediction (SHAP values)
- Additional insights:
  - Engagement score
  - High-risk employee percentages
  - Attrition trends by department, role, and tenure

---

## Tools and Technologies
- **Python**: pandas, numpy, scikit-learn, imbalanced-learn, shap, lifelines
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Model Persistence**: joblib
- **Database Integration**: SQL example provided
- **Deployment**: Streamlit Cloud or Docker

---

## How to Use the App

1. **Run the App**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
Upload Dataset

Use the sidebar to upload your IBM HR Attrition CSV file.

Alternatively, check the Use sample synthetic data option for testing.

Explore the Data

Go to the Data Summary tab to preview data and check missing values.

Use EDA tab for interactive charts, distributions, correlation heatmaps, and cohort analysis.

Survival Analysis

View Kaplan-Meier curves in the Survival Analysis tab.

Analyze employee tenure and attrition probability by department.

Train Models

Navigate to Modeling & Eval tab.

Select model type (Random Forest or Logistic Regression) and imbalance handling (None, class_weight, SMOTE).

Set test size and cross-validation folds, then click Train Model.

View evaluation metrics, confusion matrix, and feature importance.

Explain Predictions

Go to Explain & Predict tab.

View SHAP summary plot for global feature importance.

Use individual prediction form to input employee data and see:

Predicted attrition probability

Predicted class (Yes/No)

SHAP force plot showing top contributing features

Bulk Predictions

Click Predict Attrition for Filtered Dataset to get predictions for all employees in the dataset.

Download predictions as CSV.

Save/Load Model

Use Save last trained model to persist models for later use.

Load a saved .pkl model via the sidebar to perform predictions without retraining.

SQL Integration

Example SQL snippet is provided at the bottom of the app for inserting predictions into a database.
