import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

sns.set_theme()
warnings.filterwarnings("ignore")

# Create output directory for images
import os

output_dir = "images/validate-and-test-model"
os.makedirs(output_dir, exist_ok=True)

# Load data and model
data = pd.read_csv("data/raw/Time-Series_Network_logs.csv")
model = joblib.load("models/intrusion_detection_model.pkl")

# Assuming preprocessed X_test, y_test from opti-hyper-param.py logic
# Replicate necessary preprocessing here for completeness
data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["hour"] = data["Timestamp"].dt.hour
data["day_of_week"] = data["Timestamp"].dt.dayofweek
data["day"] = data["Timestamp"].dt.day
data["month"] = data["Timestamp"].dt.month
data["minute"] = data["Timestamp"].dt.minute
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
data["is_business_hours"] = ((data["hour"] >= 9) & (data["hour"] <= 17)).astype(int)
data = data.sort_values("Timestamp").reset_index(drop=True)
data["payload_rolling_mean"] = (
    data["Payload_Size"].rolling(window=10, min_periods=1).mean()
)
data["payload_rolling_std"] = (
    data["Payload_Size"].rolling(window=10, min_periods=1).std().fillna(0)
)
data["payload_lag_1"] = (
    data["Payload_Size"].shift(1).fillna(data["Payload_Size"].mean())
)
data["payload_lag_5"] = (
    data["Payload_Size"].shift(5).fillna(data["Payload_Size"].mean())
)
data["time_since_last"] = (
    data.groupby("Request_Type")["Timestamp"].diff().dt.total_seconds().fillna(0)
)

categorical_columns = ["Request_Type", "Protocol", "User_Agent", "Status"]
encoders = joblib.load("models/encoders.pkl")
for col in categorical_columns:
    data[col + "_encoded"] = encoders[col].transform(data[col])

feature_columns = [
    "Payload_Size",
    "hour",
    "day_of_week",
    "day",
    "month",
    "minute",
    "is_weekend",
    "is_business_hours",
    "payload_rolling_mean",
    "payload_rolling_std",
    "payload_lag_1",
    "payload_lag_5",
    "time_since_last",
] + [col + "_encoded" for col in categorical_columns]
X = data[feature_columns]
y = data["Intrusion"]
split_index = int(len(data) * 0.8)
X_test = X[split_index:]
y_test = y[split_index:]

y_prob = model.predict_proba(X_test)[:, 1]

# Density Chart
plt.figure(figsize=(16, 10))
sns.histplot(y_prob[y_test == 0], alpha=0.5)
plt.axvline(
    np.median(y_prob[y_test == 0]), 0, 1, linestyle="--", label="Median Class 0"
)
plt.axvline(np.mean(y_prob[y_test == 0]), 0, 1, linestyle="-", label="Mean Class 0")
sns.histplot(y_prob[y_test == 1], color="darkorange", alpha=0.4)
plt.axvline(
    np.median(y_prob[y_test == 1]),
    0,
    1,
    linestyle="--",
    color="darkorange",
    label="Median Class 1",
)
plt.axvline(
    np.mean(y_prob[y_test == 1]),
    0,
    1,
    linestyle="-",
    color="darkorange",
    label="Mean Class 1",
)
plt.legend()
plt.xlabel("Predicted probabilities")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.xlim(-0.05, 1.05)
plt.title("Density Chart", fontsize=16)
plt.savefig(f"{output_dir}/density_chart.png")
plt.show()

# Calibration Curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_prob, n_bins=20
)
plt.figure(figsize=(16, 10))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.6)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.ylabel("Fraction of positives")
plt.xlabel("Predicted probabilities")
plt.legend()
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.title("Calibration Curve")
plt.savefig(f"{output_dir}/calibration_curve_val.png")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(16, 10))
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label="ROC curve (area = {:2.1f}%)".format(auc(fpr, tpr) * 100),
)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.title("ROC Curve", fontsize=16)
plt.legend(loc="lower right")
plt.savefig(f"{output_dir}/roc_curve.png")
plt.show()

# PR Curve
plt.figure(figsize=(16, 11))
prec, recall, _ = precision_recall_curve(y_test, y_prob)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot(ax=plt.gca())
plt.title("PR Curve", fontsize=16)
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.savefig(f"{output_dir}/pr_curve.png")
plt.show()

# SHAP Interpretations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
# For binary classification, shap_values is a list with two arrays
if isinstance(shap_values, list):
    shap_values_positive = shap_values[1]  # Positive class SHAP values
else:
    shap_values_positive = shap_values

shap.summary_plot(shap_values_positive, X_test, plot_size=0.8)
plt.savefig(f"{output_dir}/shap_summary.png")
plt.show()

# Example local interpretation
print("Local SHAP interpretation for first sample:")
print("SHAP values:", shap_values_positive[0, :])
print("Feature values:", X_test.iloc[0, :].values)

# Post-training Tests: Invariance (adapt to a feature, e.g., Payload_Size)
X_test_payload = X_test[X_test["Payload_Size"] > 1]
X_test_plus = X_test_payload.copy()
X_test_plus["Payload_Size"] += 1
X_test_minus = X_test_payload.copy()
X_test_minus["Payload_Size"] -= 1
y_payload = pd.DataFrame()
y_payload["y"] = model.predict_proba(X_test_payload)[:, 1]
y_payload["y+"] = model.predict_proba(X_test_plus)[:, 1]
y_payload["y-"] = model.predict_proba(X_test_minus)[:, 1]
y_payload["abs_delta"] = np.abs(y_payload["y-"] - y_payload["y+"])
print("Invariance Test - Std of abs_delta:", y_payload["abs_delta"].std())
print("Proportion with abs_delta < 0.05:", (y_payload["abs_delta"] < 0.05).mean())

# Directional Test (adapt to features like hour or payload_lag_1)
x_unit = X_test.iloc[0].copy()
print("Original Prob:", model.predict_proba([x_unit])[:, 1])
x_unit["hour"] += 1  # Example perturbation
print("Perturbed Prob:", model.predict_proba([x_unit])[:, 1])

# Model Unit Tests with Prototypes
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_test)
X_prototypes = pd.DataFrame(data=kmeans.cluster_centers_, columns=X_test.columns)
print("Prototype Predictions:", model.predict_proba(kmeans.cluster_centers_))
