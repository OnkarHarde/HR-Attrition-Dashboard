# explainer.py
import shap
import numpy as np
import pandas as pd

class SHAPExplainerWrapper:
    def __init__(self, pipeline, model, background_df=None):
        self.pipeline = pipeline
        self.model = model

        # Auto-build background if not provided
        if background_df is None or len(background_df) == 0:
            raise ValueError("A background dataset (train_df) must be passed for SHAP.")

        transformed = pipeline.named_steps["preprocessor"].transform(background_df)
        self.background = shap.sample(transformed, 200)

        self.explainer = shap.Explainer(model.predict_proba, self.background)

    def explain(self, input_df: pd.DataFrame):
        transformed = self.pipeline.named_steps["preprocessor"].transform(input_df)
        shap_values = self.explainer(transformed)
        return shap_values
