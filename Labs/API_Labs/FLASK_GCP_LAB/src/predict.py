import numpy as np
import joblib
import os
from train import run_training

# Load the trained model
model = joblib.load("model/model.pkl")

def predict_cancer(features):
    """
    Predict breast cancer diagnosis.
    
    Args:
        features: list or array of 30 numeric features
    
    Returns:
        prediction (int): 0 = malignant, 1 = benign
    """
    input_data = np.array([features])
    prediction = model.predict(input_data)
    return prediction[0]

if __name__ == "__main__":
    if os.path.exists("model/model.pkl"):
        print("Model loaded successfully")
    else:
        os.makedirs("model", exist_ok=True)
        run_training()
        