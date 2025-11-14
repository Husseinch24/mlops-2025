from .base_model import BaseModel
import pickle
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
