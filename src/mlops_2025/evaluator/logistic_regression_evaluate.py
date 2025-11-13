from .base_evaluate import BaseEvaluator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
 
class LogistiRegressionEvalute(BaseEvaluator):
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
   
    def evaluate(self, X_test, y_test) -> dict:
        y_pred = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        }