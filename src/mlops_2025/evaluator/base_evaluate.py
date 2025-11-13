from abc import ABC, abstractmethod
import pandas as pd
 
class BaseEvaluator(ABC):
    """Abstract base class for model evaluation."""
   
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
        """
        Evaluates the model on test data.
        Returns a dictionary of metrics (e.g., accuracy, precision, recall, F1).
        """
        pass
 