# explainer.py
# SHAP explainability utilities for global and local explanations in Streamlit
# Updated: Safe background handling, fallback logic, stable SHAP usage

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SHAPExplainerWrapper:
    """
    Robust SHAP wrapper for Streamlit HR Attrition Dashboard.
    Automatically handles:
        - background dataset
        - tree & linear models
        - kernel fallback
    """

    def __init__(self, pipeline, model, background_data: pd.DataFrame = None):
        self.pipeline = pipeline
        self.model = model

        # --- Safe background handling ---
        if background_data is None or background_data.empty:
            # fallback: synthetic mean row
            cols = pipeline.named_steps["preprocessor"].get_feature_names_out()
            background_data = pd.DataFrame([np.zeros(len(cols))], columns=cols)
            self.X_background = background_data
        else:
            # preprocess background
            self.X_background = pd.DataFrame(
                pipeline.named_steps["preprocessor"].transform(background_data),
                columns=pipeline.named_steps["preprocessor"].get_feature_names_out()
            )

        # --- Choose SHAP Explainer ---
        if hasattr(model, "feature_importances_"):
            # Tree Models
            self.explainer = shap.TreeExplainer(model, self.X_background)
        else:
            # Linear Models → KernelExplainer
            self.explainer = shap.KernelExplainer(
                self._model_predict_proba,
                shap.sample(self.X_background, min(30, len(self.X_background)))
            )

    def _model_predict_proba(self, X):
        """Return probability of positive class."""
        try:
            return self.model.predict_proba(X)[:, 1]
        except:
            return self.model.predict(X)

    def summary_plot(self, X_sample=None, show=True):
        """Global SHAP summary plot."""
        if X_sample is not None:
            X_proc = pd.DataFrame(
                self.pipeline.named_steps["preprocessor"].transform(X_sample),
                columns=self.pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
        else:
            X_proc = self.X_background

        shap_values = self.explainer(X_proc)
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(vals, X_proc, show=False)
        return fig

    def force_plot(self, X_row):
        """Local SHAP force plot for a single employee."""
        X_proc = pd.DataFrame(
            self.pipeline.named_steps["preprocessor"].transform(X_row),
            columns=self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

        shap_values = self.explainer(X_proc)
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        fig = plt.figure(figsize=(10, 3))
        shap.plots.force(vals, matplotlib=True, show=False)
        return fig
