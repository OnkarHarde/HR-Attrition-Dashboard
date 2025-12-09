# explainer.py
# SHAP explainability utilities for global and local explanations in Streamlit

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class SHAPExplainerWrapper:
    """
    Wrapper around SHAP to provide global summary and local force plots.
    Works with scikit-learn pipelines and both tree and linear models.
    """

    def __init__(self, pipeline, model, background_data: pd.DataFrame = None):
        """
        pipeline: scikit-learn pipeline with 'preprocessor'
        model: fitted scikit-learn model (supports predict_proba)
        background_data: pd.DataFrame subset of training or filtered data (raw, not preprocessed)
        """
        self.pipeline = pipeline
        self.model = model

        if background_data is None or background_data.empty:
            raise ValueError("Provide a sample dataframe for SHAP background (training or subset).")

        # Preprocess background
        self.X_background = pd.DataFrame(
            self.pipeline.named_steps["preprocessor"].transform(background_data),
            columns=self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

        # Initialize appropriate SHAP explainer
        if hasattr(self.model, "feature_importances_"):
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model, data=self.X_background)
        else:
            # Linear/logistic or other models
            self.explainer = shap.KernelExplainer(
                self._model_predict_proba,
                shap.sample(self.X_background, min(50, len(self.X_background)))
            )

    def _model_predict_proba(self, X):
        """Helper for KernelExplainer: returns probability of positive class."""
        try:
            return self.model.predict_proba(X)[:, 1]
        except Exception:
            return self.model.predict(X)

    def summary_plot(self, X_sample: pd.DataFrame = None, show=True):
        """
        Global SHAP summary.
        X_sample: raw dataframe (not preprocessed). Defaults to background.
        """
        if X_sample is not None:
            X_proc = pd.DataFrame(
                self.pipeline.named_steps["preprocessor"].transform(X_sample),
                columns=self.pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
        else:
            X_proc = self.X_background

        # Compute SHAP values
        shap_values = self.explainer(X_proc)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # binary classification
        else:
            shap_vals = shap_values

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_proc, show=show)
        return fig

    def force_plot(self, X_row: pd.DataFrame):
        """
        Local SHAP force plot for one employee.
        X_row: raw 1-row DataFrame (not preprocessed)
        """
        X_proc = pd.DataFrame(
            self.pipeline.named_steps["preprocessor"].transform(X_row),
            columns=self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

        shap_values = self.explainer(X_proc)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # first row for binary classification
        else:
            shap_vals = shap_values[0]

        fig = plt.figure(figsize=(10, 3))
        shap.plots.force(shap_vals, matplotlib=True, show=False)
        return fig
