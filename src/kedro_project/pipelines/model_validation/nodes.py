import logging
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm.sklearn import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Set seaborn theme
sns.set_theme()


def generate_probability_predictions(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Generate probability predictions for validation."""

    y_prob = model.predict_proba(X_test)[:, 1]

    predictions_data = {
        "y_test": y_test.tolist(),
        "y_prob": y_prob.tolist(),
        "test_size": len(X_test),
        "intrusion_rate": float(y_test.mean()),
    }

    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Intrusion rate in test: {y_test.mean():.2%}")

    return predictions_data


def create_density_chart(predictions_data: dict[str, Any]) -> plt.Figure:
    """Create density chart visualization."""

    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot histograms
    sns.histplot(y_prob[y_test == 0], alpha=0.5, label="Normal", color="blue", ax=ax)
    ax.axvline(
        np.median(y_prob[y_test == 0]),
        0,
        1,
        linestyle="--",
        color="blue",
        label="Median Normal",
    )
    ax.axvline(
        np.mean(y_prob[y_test == 0]),
        0,
        1,
        linestyle="-",
        color="blue",
        label="Mean Normal",
    )

    sns.histplot(
        y_prob[y_test == 1], color="darkorange", alpha=0.4, label="Intrusion", ax=ax
    )
    ax.axvline(
        np.median(y_prob[y_test == 1]),
        0,
        1,
        linestyle="--",
        color="darkorange",
        label="Median Intrusion",
    )
    ax.axvline(
        np.mean(y_prob[y_test == 1]),
        0,
        1,
        linestyle="-",
        color="darkorange",
        label="Mean Intrusion",
    )

    ax.legend()
    ax.set_xlabel("Predicted Probabilities")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Density Chart - Class Distributions", fontsize=16)

    plt.tight_layout()
    return fig


def create_calibration_curve(predictions_data: dict[str, Any]) -> plt.Figure:
    """Create calibration curve visualization."""

    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=20
    )

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.6)
    ax.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        label="Intrusion Detection Model",
        color="darkorange",
        markersize=8,
    )

    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted probability")
    ax.legend()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.set_title("Calibration Curve - Intrusion Detection Model")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_roc_curve(predictions_data: dict[str, Any]) -> plt.Figure:
    """Create ROC curve visualization."""

    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=3,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.6)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.set_title("ROC Curve - Intrusion Detection", fontsize=16)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_pr_curve(predictions_data: dict[str, Any]) -> plt.Figure:
    """Create Precision-Recall curve visualization."""

    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fig, ax = plt.subplots(figsize=(16, 11))

    prec, recall, _ = precision_recall_curve(y_test, y_prob)
    PrecisionRecallDisplay(precision=prec, recall=recall).plot(
        ax=ax, color="darkorange", lw=3
    )

    ax.set_title("Precision-Recall Curve - Intrusion Detection", fontsize=16)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_invariance_test(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    test_feature: str = "Payload_Size",
) -> dict[str, Any]:
    """Run invariance test on model predictions."""

    # Check if feature exists
    if test_feature not in X_test.columns:
        raise ValueError(f"Feature '{test_feature}' not found in test data")

    # Filter test data
    X_test_filtered = X_test[X_test[test_feature] > 1]

    # Check if there's enough data after filtering
    if len(X_test_filtered) == 0:
        raise ValueError(f"No data remains after filtering for feature '{test_feature}' > 1")

    # Create perturbed versions
    X_test_plus = X_test_filtered.copy()
    X_test_plus[test_feature] += 1

    X_test_minus = X_test_filtered.copy()
    X_test_minus[test_feature] -= 1

    # Get predictions
    y_plus = model.predict_proba(X_test_plus)[:, 1]
    y_minus = model.predict_proba(X_test_minus)[:, 1]

    # Calculate invariance metrics
    abs_delta = np.abs(y_minus - y_plus)

    invariance_results = {
        "abs_delta_std": abs_delta.std(),
        "proportion_stable": (abs_delta < 0.05).mean(),
        "test_feature": test_feature,
        "test_samples": len(X_test_filtered),
    }

    logger.info(f"Invariance Test Results for {test_feature}:")
    logger.info(f"  Std of abs_delta: {invariance_results['abs_delta_std']:.6f}")
    logger.info(
        f"  Proportion with abs_delta < 0.05: {invariance_results['proportion_stable']:.3f}"
    )

    return invariance_results


def run_prototype_test(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    n_clusters: int = 10,
) -> dict[str, Any]:
    """Run prototype test using K-means clustering."""

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_test)

    # Get prototype predictions
    prototype_predictions = model.predict_proba(kmeans.cluster_centers_)[:, 1]

    prototype_results = {
        "n_clusters": n_clusters,
        "prototype_predictions": prototype_predictions.tolist(),
        "cluster_centers": kmeans.cluster_centers_.tolist(),
    }

    logger.info(f"Prototype Test Results ({n_clusters} clusters):")
    logger.info(f"  Prototype predictions: {prototype_predictions}")

    return prototype_results
