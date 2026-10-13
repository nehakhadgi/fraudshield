from statistics import fmean, pstdev

import streamlit as st

from model import FEATURE_NAMES, build_risk_model, split_dataset

st.set_page_config(page_title="FraudShield ML System", layout="wide")


@st.cache_resource
def load_model():
    X_train, _, y_train, _ = split_dataset()
    model = build_risk_model()
    model.fit(X_train, y_train)
    reference_rows = [list(row) for row in X_train]
    return model, reference_rows
model, reference_rows = load_model()

st.title("FraudShield ML System")
st.write(
    "Synthetic demonstration only. The signals and decisions are not suitable "
    "for real financial or fraud-control use."
)

score_tab, monitor_tab = st.tabs(["Score transaction", "Monitor drift"])
with score_tab:
    values = []
    for feature_name in FEATURE_NAMES:
        values.append(st.slider(feature_name, -5.0, 5.0, 0.0, 0.1))

    threshold = st.slider("Manual review threshold", 0.05, 0.95, 0.50, 0.05)
    risk = float(model.predict_proba([values])[0][1])

    st.metric("Estimated fraud risk", f"{risk:.1%}")
    if risk >= threshold:
        st.error("Send this transaction for manual review.")
    else:
        st.success("Approve this transaction under the current threshold.")

    st.write(
        "Move the threshold to see how an operational policy changes the decision "
        "without retraining the model."
    )
def calculate_drift(reference_rows, batch_rows):
    scores = []
    for index in range(len(FEATURE_NAMES)):
        reference_values = [row[index] for row in reference_rows]
        batch_values = [row[index] for row in batch_rows]
        score = abs(fmean(batch_values) - fmean(reference_values)) / (pstdev(reference_values) + 1e-9)
        scores.append(score)
    return scores
with monitor_tab:
    shift_magnitude = st.slider("Feature shift magnitude", 0.0, 4.0, 0.0, 0.25)
    retraining_trigger = st.slider("Retraining trigger", 0.5, 2.0, 1.0, 0.1)

    batch_rows = [row[:] for row in reference_rows[:400]]
    for row in batch_rows:
        row[0] += shift_magnitude

    drift_scores = calculate_drift(reference_rows, batch_rows)
    drift_rows = [
        {"Feature": feature_name, "Standardized mean shift": round(float(score), 2)}
        for feature_name, score in zip(FEATURE_NAMES, drift_scores)
    ]
    maximum_drift = max(drift_scores)

    st.dataframe(drift_rows, hide_index=True, width="stretch")
    st.metric("Maximum drift score", f"{maximum_drift:.2f}")
    st.metric("Retraining trigger", f"{retraining_trigger:.2f}")

    if maximum_drift > retraining_trigger:
        st.error("Retraining trigger met: open a model review and retraining task.")
    else:
        st.success("Monitor only: the retraining trigger has not been met.")

    st.write(
        "The trigger is a project-defined policy for this synthetic demonstration, "
        "not a universal production threshold."
    )