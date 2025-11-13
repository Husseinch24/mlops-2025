from xgboost import XGBClassifier
from .base_model import BaseModel
import pickle
 
class XGBoostModel(BaseModel):
    def __init__(self):
        self.model = XGBClassifier()
 
    def train(self, X, y):
        self.model.fit(X, y)
   
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)