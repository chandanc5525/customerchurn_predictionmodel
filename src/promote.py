import shutil
import os

def promote_to_production():
    src = "models/staging_model.joblib"
    dst = "models/production_model.joblib"

    if not os.path.exists(src):
        raise FileNotFoundError("No staging model to promote!")

    shutil.copy(src, dst)
    print("Model promoted to production")