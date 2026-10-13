import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score

from model import build_risk_model, split_dataset


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = [row[1] for row in model.predict_proba(X_test)]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "fraud_recall": recall_score(y_test, predictions),
        "average_precision": average_precision_score(y_test, probabilities),
    }
X_train, X_test, y_train, y_test = split_dataset()
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-risk-classification")

experiments = [
    (
        "majority-baseline",
        DummyClassifier(strategy="most_frequent"),
        {"model_type": "DummyClassifier", "strategy": "most_frequent"},
    ),
    (
        "balanced-logistic-regression",
        build_risk_model(),
        {"model_type": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000},
    ),
]
for run_name, model, parameters in experiments:
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model, name="model")

        print(run_name)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.3f}")