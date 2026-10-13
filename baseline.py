from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, average_precision_score, recall_score

from model import split_dataset

X_train, X_test, y_train, y_test = split_dataset()
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = [row[1] for row in model.predict_proba(X_test)]

print("Majority-class baseline")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Fraud recall: {recall_score(y_test, predictions):.3f}")
print(f"Average precision: {average_precision_score(y_test, probabilities):.3f}")