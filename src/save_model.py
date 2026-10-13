# New imports
import os
from joblib import dump
from sklearn.pipeline import Pipeline


# New function
def save_model(model: Pipeline, model_path: str) -> None:
    """
    Saves the trained model to a file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    dump(model, model_path)
    print(f"Saved model to {model_path}")