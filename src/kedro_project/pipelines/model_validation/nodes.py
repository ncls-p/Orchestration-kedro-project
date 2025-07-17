from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm.sklearn import LGBMClassifier
from scipy import sparse
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# --------------------------------------------------------------------
# Global threshold for "stable" delta (replaces 0.05 literals, PLR2004)
# --------------------------------------------------------------------
STABLE_DELTA_THRESHOLD: float = 0.05

logger = logging.getLogger(__name__)


def _get_positive_class_probabilities(y_prob_matrix: Any) -> np.ndarray:
    """Extract positive class probabilities from prediction matrix.

    Args:
        y_prob_matrix: Probability matrix from model.predict_proba()

    Returns:
        Array of positive class probabilities
    """
    if sparse.issparse(y_prob_matrix):
        y_prob_matrix = y_prob_matrix.toarray()
    return y_prob_matrix[:, 1]


__all__ = [
    "evaluate_model",
    "generate_probability_predictions",
    "create_density_chart",
    "create_calibration_curve",
    "create_roc_curve",
    "create_pr_curve",
    "run_invariance_test",
    "run_prototype_test",
]

# Set seaborn theme
sns.set_theme()


def generate_probability_predictions(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Generate probability predictions for validation.

    Args:
        model: The trained LightGBM classifier
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict containing test predictions and metadata:
            - y_test: Test labels as list
            - y_prob: Predicted probabilities as list
            - test_size: Size of test set
            - intrusion_rate: Rate of intrusions in test set

    Examples:
        >>> predictions = generate_probability_predictions(model, X_test, y_test)
        >>> print(f"Test size: {predictions['test_size']}")
    """
    y_prob = _get_positive_class_probabilities(model.predict_proba(X_test))

    predictions_data = {
        "y_test": y_test.tolist(),
        "y_prob": y_prob.tolist(),
        "test_size": len(X_test),
        "intrusion_rate": float(y_test.mean()),
    }

    logger.info(f"Test set size: {len(X_test)}")
    logger.info(f"Intrusion rate in test: {y_test.mean():.2%}")

    return predictions_data


def create_density_chart(predictions_data: dict[str, Any]) -> Any:
    """
    Create density chart visualization.

    Args:
        predictions_data: Dictionary containing prediction data with keys:
            - y_test: Test labels
            - y_prob: Predicted probabilities

    Returns:
        matplotlib.figure.Figure: Density chart visualization

    Examples:
        >>> predictions = {"y_test": [0, 1, 0], "y_prob": [0.2, 0.8, 0.1]}
        >>> fig = create_density_chart(predictions)
        >>> fig.show()
    """
    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot histograms
    sns.histplot(y_prob[y_test == 0], alpha=0.5, label="Normal", color="blue", ax=ax)
    ax.axvline(
        float(np.median(y_prob[y_test == 0])),
        0,
        1,
        linestyle="--",
        color="blue",
        label="Median Normal",
    )
    ax.axvline(
        float(np.mean(y_prob[y_test == 0])),
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
        float(np.median(y_prob[y_test == 1])),
        0,
        1,
        linestyle="--",
        color="darkorange",
        label="Median Intrusion",
    )
    ax.axvline(
        float(np.mean(y_prob[y_test == 1])),
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


def create_calibration_curve(predictions_data: dict[str, Any]) -> Any:
    """
    Create calibration curve visualization.

    Args:
        predictions_data: Dictionary containing prediction data with keys:
            - y_test: Test labels
            - y_prob: Predicted probabilities

    Returns:
        matplotlib.figure.Figure: Calibration curve visualization

    Examples:
        >>> predictions = {"y_test": [0, 1, 0], "y_prob": [0.2, 0.8, 0.1]}
        >>> fig = create_calibration_curve(predictions)
        >>> fig.show()
    """
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


def create_roc_curve(predictions_data: dict[str, Any]) -> Any:
    """
    Create ROC curve visualization.

    Args:
        predictions_data: Dictionary containing prediction data with keys:
            - y_test: Test labels
            - y_prob: Predicted probabilities

    Returns:
        matplotlib.figure.Figure: ROC curve visualization

    Examples:
        >>> predictions = {"y_test": [0, 1, 0], "y_prob": [0.2, 0.8, 0.1]}
        >>> fig = create_roc_curve(predictions)
        >>> fig.show()
    """
    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Log ROC AUC to MLflow
    mlflow.log_metric("roc_auc", roc_auc)

    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=3,
        label=f"ROC curve (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.6)

    ax.set_xlim((-0.01, 1.01))
    ax.set_ylim((-0.01, 1.01))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    ax.set_title("ROC Curve - Intrusion Detection", fontsize=16)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_pr_curve(predictions_data: dict[str, Any]) -> Any:
    """
    Create Precision-Recall curve visualization.

    Args:
        predictions_data: Dictionary containing prediction data with keys:
            - y_test: Test labels
            - y_prob: Predicted probabilities

    Returns:
        matplotlib.figure.Figure: Precision-Recall curve visualization

    Examples:
        >>> predictions = {"y_test": [0, 1, 0], "y_prob": [0.2, 0.8, 0.1]}
        >>> fig = create_pr_curve(predictions)
        >>> fig.show()
    """
    y_test = np.array(predictions_data["y_test"])
    y_prob = np.array(predictions_data["y_prob"])

    fig, ax = plt.subplots(figsize=(16, 11))

    prec, recall, _ = precision_recall_curve(y_test, y_prob)

    # Calculate and log PR AUC
    pr_auc = auc(recall, prec)
    mlflow.log_metric("pr_auc", pr_auc)

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
        raise ValueError(
            f"No data remains after filtering for feature '{test_feature}' > 1"
        )

    # Create perturbed versions
    X_test_plus = X_test_filtered.copy()
    X_test_plus[test_feature] += 1

    X_test_minus = X_test_filtered.copy()
    X_test_minus[test_feature] -= 1

    # Get predictions
    y_plus = _get_positive_class_probabilities(model.predict_proba(X_test_plus))
    y_minus = _get_positive_class_probabilities(model.predict_proba(X_test_minus))

    # Calculate invariance metrics
    abs_delta = np.abs(y_minus - y_plus)

    invariance_results = {
        "abs_delta_std": abs_delta.std(),
        "proportion_stable": (abs_delta < STABLE_DELTA_THRESHOLD).mean(),
        "test_feature": test_feature,
        "test_samples": len(X_test_filtered),
    }

    logger.info(f"Invariance Test Results for {test_feature}:")
    logger.info(f"  Std of abs_delta: {invariance_results['abs_delta_std']:.6f}")
    logger.info(
        f"  Proportion with abs_delta < {STABLE_DELTA_THRESHOLD}: "
        f"{invariance_results['proportion_stable']:.3f}"
    )

    return invariance_results


def run_prototype_test(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    n_clusters: int = 10,
) -> dict[str, Any]:
    """
    Run prototype test using K-means clustering.

    Args:
        model: The trained LightGBM classifier
        X_test: Test features
        n_clusters: Number of clusters for K-means

    Returns:
        Dict containing prototype test results:
            - n_clusters: Number of clusters used
            - prototype_predictions: Predictions for cluster centers
            - cluster_centers: Cluster centers coordinates

    Examples:
        >>> results = run_prototype_test(model, X_test, n_clusters=5)
        >>> print(f"Used {results['n_clusters']} clusters")
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_test)

    # Get prototype predictions
    prototype_predictions = _get_positive_class_probabilities(
        model.predict_proba(kmeans.cluster_centers_)
    )

    prototype_results = {
        "n_clusters": n_clusters,
        "prototype_predictions": prototype_predictions.tolist(),
        "cluster_centers": kmeans.cluster_centers_.tolist(),
    }

    logger.info(f"Prototype Test Results ({n_clusters} clusters):")
    logger.info(f"  Prototype predictions: {prototype_predictions}")

    return prototype_results


def _compute_metrics(
    model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series, cv: TimeSeriesSplit
) -> dict[str, Any]:
    """
    Compute evaluation metrics for the model.

    Args:
        model: The trained model to evaluate
        X_test: Test features
        y_test: Test labels
        cv: Cross-validation splitter

    Returns:
        Dict containing raw metrics and predictions

    Raises:
        ValueError: If model cannot make predictions

    Examples:
        >>> cv = TimeSeriesSplit(n_splits=5)
        >>> metrics = _compute_metrics(model, X_test, y_test, cv)
        >>> print(metrics["accuracy"])
    """
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="accuracy")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = _get_positive_class_probabilities(model.predict_proba(X_test))

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "cv_scores": cv_scores,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def _aggregate_metrics(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate and format metrics for reporting.

    Args:
        raw_metrics: Dictionary containing raw metrics from _compute_metrics

    Returns:
        Dict containing aggregated metrics ready for reporting

    Examples:
        >>> raw_metrics = _compute_metrics(model, X_test, y_test, cv)
        >>> aggregated = _aggregate_metrics(raw_metrics)
        >>> print(aggregated["cv_mean_accuracy"])
    """
    cv_scores = raw_metrics["cv_scores"]

    return {
        "accuracy": raw_metrics["accuracy"],
        "precision": raw_metrics["precision"],
        "recall": raw_metrics["recall"],
        "f1_score": raw_metrics["f1_score"],
        "roc_auc": raw_metrics["roc_auc"],
        "confusion_matrix": raw_metrics["confusion_matrix"].tolist(),
        "cv_scores": cv_scores.tolist(),
        "cv_mean_accuracy": cv_scores.mean(),
        "cv_std_accuracy": cv_scores.std(),
    }


def _log_metrics(metrics: dict[str, Any]) -> None:
    """
    Log metrics to MLflow tracking system.

    Args:
        metrics: Dictionary containing metrics to log

    Examples:
        >>> metrics = {"accuracy": 0.95, "f1_score": 0.94}
        >>> _log_metrics(metrics)
    """
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)
        elif isinstance(metric_value, list):
            # Log list metrics as individual items
            for i, value in enumerate(metric_value):
                mlflow.log_metric(f"{metric_name}_{i}", value)
        else:
            # Log other types as text
            mlflow.log_text(str(metric_value), f"{metric_name}.txt")


def evaluate_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """
    Evaluate model performance using cross-validation and various metrics.

    Args:
        model: The trained LightGBM classifier
        X_test: Test features
        y_test: Test labels
        cv_folds: Number of cross-validation folds

    Returns:
        Dict containing comprehensive evaluation metrics

    Examples:
        >>> model = LGBMClassifier()
        >>> X_test = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> y_test = pd.Series([0, 1, 0])
        >>> results = evaluate_model(model, X_test, y_test)
        >>> print(results["accuracy"])
    """
    logger.info("Starting model evaluation...")
    cv = TimeSeriesSplit(n_splits=cv_folds)

    # Compute raw metrics
    raw_metrics = _compute_metrics(model, X_test, y_test, cv)

    # Aggregate metrics for reporting
    metrics = _aggregate_metrics(raw_metrics)

    # Log metrics to MLflow
    _log_metrics(metrics)

    logger.info("Model evaluation completed successfully")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")

    return metrics
