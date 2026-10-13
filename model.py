from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "Transaction velocity signal",
    "Amount anomaly signal",
    "Merchant risk signal",
    "Device mismatch signal",
]


def make_dataset():
    return make_classification(
        n_samples=4000,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.98, 0.02],
        flip_y=0.01,
        class_sep=0.8,
        random_state=42,
    )
def split_dataset():
    X, y = make_dataset()
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
def build_risk_model():
    return LogisticRegression(class_weight="balanced", max_iter=1000)