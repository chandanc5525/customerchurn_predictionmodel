import joblib
import os

def validate_staging():
    path = "models/staging_model.joblib"
    if not os.path.exists(path):
        raise FileNotFoundError("Staging model not found. Train first!")

    model = joblib.load(path)
    print("Staging model validated")
    return True
