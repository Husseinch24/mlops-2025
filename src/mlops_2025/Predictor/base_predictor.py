from abc import ABC, abstractmethod
import pandas as pd
 
class BasePredictor(ABC):
    """Abstract base class for making predictions."""
   
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Makes predictions on the input data.
        Returns a DataFrame of predictions.
        """
        pass 