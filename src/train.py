
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path


# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)


# 2. Split into training and testing data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Configure training-data fraction
# Use 1.0 for the baseline.
# Try 0.2 for the reduced-data experiment.
TRAIN_DATA_FRACTION = 0.5

X_train, _, y_train, _ = train_test_split(
    X_train_full,
    y_train_full,
    train_size=TRAIN_DATA_FRACTION,
    random_state=42,
    stratify=y_train_full
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Create model
model = LogisticRegression(
    max_iter=5000,
    random_state=42
)


# 5. Configure MLflow
PROJECT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_DIR / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("ML_CI_CD_Demo")


# 6. Train, evaluate, and track
with mlflow.start_run():

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 5000)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("train_data_fraction", TRAIN_DATA_FRACTION)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", round(accuracy, 4))

    mlflow.log_metric("accuracy", accuracy)

    # 7. Model quality gate
    MIN_ACCURACY = 0.90

    mlflow.log_metric("minimum_required_accuracy", MIN_ACCURACY)

    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Model quality gate failed! "
            f"Accuracy={accuracy:.4f}, "
            f"Required={MIN_ACCURACY:.2f}"
        )

    print(f"Model quality gate passed: {accuracy:.4f}")


    # 8. Save model only if quality gate passes
    models_dir = PROJECT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


    # 9. Log model to MLflow
    mlflow.sklearn.log_model(model, "model")

    print("Model logged to MLflow")