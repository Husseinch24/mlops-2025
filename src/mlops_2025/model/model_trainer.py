import pandas as pd

from mlops_2025.model.xgboost import XGBoostModel
from .logistic_regression_model import LogisticRegressionModel
from .random_forest_model import RandomForestModel  # if you have it
# from .xgboost_model import XGBoostModel  # optional
from typing import Union

class ModelTrainer:
    """Trainer class that wraps different model implementations."""

    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.model = self._select_model(self.model_name)

    def _select_model(self, model_name: str):
        if model_name == "logreg":
            return LogisticRegressionModel()
        elif model_name == "rf":
            return RandomForestModel()
        elif model_name == "xgb":
            return XGBoostModel()
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    # model_trainer.py
    def train(self, X, y):
        """Train model given features X and target y"""
        self.model.train(X, y)  # No need to drop anything


    def save_model(self, path: str):
        """Save the trained model to disk."""
        self.model.save(path)
