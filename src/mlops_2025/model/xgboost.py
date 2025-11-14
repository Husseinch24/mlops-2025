from .base_model import BaseModel
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score

class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    def train(self, X, y):
        """Train the XGBoost model."""
        self.model.fit(X, y)
    
    def save(self, path: str):
        """Save the trained XGBoost model."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, X):
        """Predict using the trained XGBoost model."""
        return self.model.predict(X)

    def predict(self, X):
        """Predict using the trained XGBoost model."""
        return self.model.predict(X)
