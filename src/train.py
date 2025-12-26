import os
import mlflow
import joblib
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from data import load_data
from feature import make_features


def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def train_all():
    os.makedirs("models", exist_ok=True)

    df, cfg = load_data()
    X_train, X_test, y_train, y_test = make_features(df, cfg["target"])

    preprocessor = build_preprocessor(X_train)

    mlflow.set_experiment(cfg["experiment"])

    best_score = 0.0
    best_model = None
    best_name = None

    for name, model in MODELS.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)
            score = f1_score(y_test, preds)

            mlflow.log_metric("f1", score)
            mlflow.log_param("model_name", name)
            mlflow.sklearn.log_model(pipe, artifact_path="model")

            print(f"Model: {name}, F1: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = pipe
                best_name = name

    if best_model is None:
        raise RuntimeError("No model trained successfully!")

    joblib.dump(best_model, "models/staging_model.joblib")
    print(f"Best model '{best_name}' saved to staging with F1 = {best_score:.4f}")


if __name__ == "__main__":
    train_all()
