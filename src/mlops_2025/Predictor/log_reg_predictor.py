from .base_predictor import BasePredictor
import pickle
 
class LogRegPredictor(BasePredictor):
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
   
    def predict(self, X):
        return self.model.predict(X)