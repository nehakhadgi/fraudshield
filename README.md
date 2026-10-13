# FraudShield: Interactive ML Model Deployment & Drift Monitoring

FraudShield is an end-to-end machine learning system designed to detect and monitor fraudulent financial transactions. The system trains a classification model, tracks performance metrics across different operational iterations using MLflow, and hosts an interactive Streamlit dashboard featuring live scoring simulations and real-time data drift monitoring.

## 🚀 Key Features

*   **Model Comparison Tracking (MLflow):** Leveraged MLflow tracking to experiment with baseline and balanced logistic regression runs, addressing extreme class imbalance to optimize model recall.
*   **Live Interactive Scoring:** Built a responsive Streamlit dashboard to simulate fraud risk scoring dynamically using feature-shifting sliders.
*   **Dynamic Decision Thresholds:** Implemented operational decision thresholds allowing users to adjust risk-approval ratios without retraining.
*   **Continuous Monitoring & Drift Detection:** Programmed a custom statistical drift-detection utility evaluating standardized mean shift between baseline datasets and incoming transaction batches.

## 🛠️ Architecture & Tech Stack

*   **Machine Learning Model:** Scikit-learn (Logistic Regression with balanced class weighting)
*   **Dashboard Framework:** Streamlit
*   **Experiment & Model Tracking:** MLflow
*   **Data Manipulation:** Pandas, NumPy
*   **Language:** Python 3.12

## 📦 Getting Started

Follow these steps to run the training pipeline, track experiments, and launch the monitoring app locally on a Windows machine.

### 1. Clone & Set Up Workspace

```bash
git clone https://github.com/nehakhadgi/fraudshield.git
cd fraudshield
```

### 2. Configure Virtual Environment & Dependencies

Create and activate a virtual environment, then install requirements:

```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Track Model Training in MLflow

Start the local MLflow server to track your experiment metrics and parameters:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Run your model script to train the algorithms and log metrics:

```bash
python model.py
```

Open `http://localhost:5000` in your browser to analyze baseline runs and comparison graphs.

### 4. Run the Monitoring Application

With your virtual environment active, run the Streamlit server:

```bash
python -m streamlit run app.py
```

Access the interactive scoring app at `http://localhost:8501`.

## 📈 Project Impact & Results

By moving from a raw majority-baseline predictor to a balanced logistic regression classifier, this system recovered critical fraud signal capture:
*   **Baseline Recall:** 0.0% (predicted the majority class)
*   **Balanced Model Recall:** Successfully improved to capture risk anomalies.
*   **Operational Readiness:** The live drift monitor flags mean-shift deviations exceeding 1.0 standard deviations to notify engineers when model performance degradation might occur due to distribution changes.
