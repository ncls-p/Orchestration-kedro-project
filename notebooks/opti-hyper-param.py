import os
import warnings

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import fmin, hp, tpe
from lightgbm.sklearn import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

sns.set_theme()
warnings.filterwarnings("ignore")

# Create output directory for images
output_dir = "images/opti-hyper-param"
os.makedirs(output_dir, exist_ok=True)

# Load the time series network logs data
data = pd.read_csv("data/raw/Time-Series_Network_logs.csv")

# Convert timestamp to datetime
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# Feature engineering for time series
data["hour"] = data["Timestamp"].dt.hour
data["day_of_week"] = data["Timestamp"].dt.dayofweek
data["day"] = data["Timestamp"].dt.day
data["month"] = data["Timestamp"].dt.month
data["minute"] = data["Timestamp"].dt.minute
data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
data["is_business_hours"] = ((data["hour"] >= 9) & (data["hour"] <= 17)).astype(int)

# Sort by timestamp to ensure proper time series order
data = data.sort_values("Timestamp").reset_index(drop=True)

# Create rolling window features (last 10 records)
data["payload_rolling_mean"] = (
    data["Payload_Size"].rolling(window=10, min_periods=1).mean()
)
data["payload_rolling_std"] = (
    data["Payload_Size"].rolling(window=10, min_periods=1).std().fillna(0)
)

# Create lag features (previous values)
data["payload_lag_1"] = (
    data["Payload_Size"].shift(1).fillna(data["Payload_Size"].mean())
)
data["payload_lag_5"] = (
    data["Payload_Size"].shift(5).fillna(data["Payload_Size"].mean())
)

# Time since last similar request (by Request_Type)
data["time_since_last"] = (
    data.groupby("Request_Type")["Timestamp"].diff().dt.total_seconds().fillna(0)
)

# Encode categorical variables
categorical_columns = [
    # "Source_IP",  # Removed - certain IPs only appear in intrusions (data leakage)
    # "Destination_IP",  # Removed - each dest IP appears only once, mostly intrusions
    "Request_Type",
    "Protocol",
    "User_Agent",
    "Status",
    # "Scan_Type",  # Removed due to data leakage - perfectly correlates with target
]

encoders = {}

for col in categorical_columns:
    encoders[col] = LabelEncoder()
    data[col + "_encoded"] = encoders[col].fit_transform(data[col])

# Prepare features and target (excluding Port due to data leakage)
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

# Split data chronologically for time series (80/20 split)
split_index = int(len(data) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Intrusion rate in training: {y_train.mean():.2%}")
print(f"Intrusion rate in test: {y_test.mean():.2%}")

MODEL_SPECS = {
    "name": "LightGBM",
    "class": LGBMClassifier,
    "max_evals": 20,
    "params": {
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.3),
        "num_iterations": hp.quniform("num_iterations", 100, 1000, 50),
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "num_leaves": hp.quniform("num_leaves", 10, 100, 10),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "min_child_samples": hp.quniform("min_child_samples", 5, 30, 5),
        "reg_alpha": hp.choice("reg_alpha", [0, 0.01, 0.1, 0.5, 1]),
        "reg_lambda": hp.choice("reg_lambda", [0, 0.01, 0.1, 0.5, 1]),
        "scale_pos_weight": hp.uniform(
            "scale_pos_weight", 1, 10
        ),  # For imbalanced data
    },
    "override_schemas": {
        "num_leaves": int,
        "min_child_samples": int,
        "max_depth": int,
        "num_iterations": int,
    },
}


def optimize_hyp(training_set, search_space, metric, evals=10):
    X_train_full, y_train_full = training_set

    def objective(params):
        # Cast parameters to correct types
        for param in set(list(MODEL_SPECS["override_schemas"].keys())).intersection(
            set(params.keys())
        ):
            cast_instance = MODEL_SPECS["override_schemas"][param]
            params[param] = cast_instance(params[param])

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores_test = []

        for train_idx, val_idx in tscv.split(X_train_full):
            X_fold_train = X_train_full.iloc[train_idx]
            y_fold_train = y_train_full.iloc[train_idx]
            X_fold_val = X_train_full.iloc[val_idx]
            y_fold_val = y_train_full.iloc[val_idx]

            # Train model
            model = LGBMClassifier(
                **params, objective="binary", verbose=-1, random_state=42
            )
            model.fit(X_fold_train, y_fold_train)

            # Evaluate on validation set
            y_pred = model.predict(X_fold_val)
            scores_test.append(metric(y_fold_val, y_pred))

        return np.mean(scores_test)

    return fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=evals,
        rstate=np.random.default_rng(42),
    )


# Optimize hyperparameters
print("\nStarting hyperparameter optimization...")
optimum_params = optimize_hyp(
    training_set=(X_train, y_train),
    search_space=MODEL_SPECS["params"],
    metric=lambda y_true, y_pred: -f1_score(y_true, y_pred),  # Minimize negative F1
    evals=MODEL_SPECS["max_evals"],
)

# Cast optimized parameters
if optimum_params is not None:
    for param in MODEL_SPECS["override_schemas"]:
        if param in optimum_params:
            cast_instance = MODEL_SPECS["override_schemas"][param]
            optimum_params[param] = cast_instance(optimum_params[param])

    print("\nOptimized parameters:")
    for key, value in optimum_params.items():
        print(f"  {key}: {value}")
else:
    print("\nOptimization failed, using default parameters")
    optimum_params = {}

# Train final model with optimized parameters
model = LGBMClassifier(
    **optimum_params, objective="binary", verbose=-1, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.asarray(y_pred)

print("\n=== Model Performance ===")
print(f"F1 Score: {f1_score(y_test, y_pred) * 100:.1f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.1f}%")
print(f"Recall: {recall_score(y_test, y_pred) * 100:.1f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Intrusion"]))

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/intrusion_detection_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")
print("\nModel saved to models/intrusion_detection_model.pkl")

prob_pos = model.predict_proba(X_test)[:, 1]  # type: ignore[index]
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, prob_pos, n_bins=10
)

# Feature importance plot
feature_importance = pd.DataFrame(
    {"feature": feature_columns, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance["feature"][:15], feature_importance["importance"][:15])
plt.xlabel("Feature Importance")
plt.title("Top 15 Most Important Features for Intrusion Detection")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png")
plt.show()

# Calibration curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.6)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
plt.ylabel("Fraction of positives")
plt.xlabel("Mean predicted probability")
plt.legend()
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
plt.title("Calibration Curve - Intrusion Detection Model")
plt.tight_layout()
plt.savefig(f"{output_dir}/calibration_curve.png")
plt.show()

# Time-based analysis
test_data = data[split_index:].copy()
test_data["predictions"] = y_pred
test_data["prob_intrusion"] = prob_pos

# Plot intrusion probability over time
plt.figure(figsize=(14, 6))
plt.scatter(
    test_data["Timestamp"],
    test_data["prob_intrusion"],
    c=test_data["Intrusion"],
    cmap="RdYlBu_r",
    alpha=0.6,
)
plt.xlabel("Timestamp")
plt.ylabel("Predicted Intrusion Probability")
plt.title("Intrusion Probability Over Time (Red=Actual Intrusion, Blue=Normal)")
plt.colorbar(label="Actual Intrusion")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/intrusion_probability_timeline.png")
plt.show()
