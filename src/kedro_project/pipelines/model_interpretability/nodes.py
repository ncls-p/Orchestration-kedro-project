import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm.sklearn import LGBMClassifier

logger = logging.getLogger(__name__)

# Set seaborn theme
sns.set_theme()


def compute_shap_values(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Compute SHAP values for model interpretability."""

    logger.info("Computing SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]  # Positive class (intrusion) SHAP values
    else:
        shap_values_positive = shap_values

    shap_data = {
        "shap_values": shap_values_positive,
        "feature_names": X_test.columns.tolist(),
        "X_test": X_test,
    }

    logger.info("SHAP values computed successfully")

    return shap_data


def create_shap_summary_plot(shap_data: dict[str, Any]) -> plt.Figure:
    """Create SHAP summary plot."""

    fig, ax = plt.subplots(figsize=(12, 8))

    shap.summary_plot(
        shap_data["shap_values"], shap_data["X_test"], plot_size=0.8, show=False
    )

    plt.title(
        "SHAP Summary Plot - Feature Importance for Intrusion Detection", fontsize=14
    )
    plt.tight_layout()

    return fig


def create_shap_feature_importance(
    shap_data: dict[str, Any],
) -> tuple[pd.DataFrame, plt.Figure]:
    """Create SHAP-based feature importance plot."""

    # Calculate feature importance from SHAP values
    feature_importance = pd.DataFrame(
        {
            "feature": shap_data["feature_names"],
            "importance": np.abs(shap_data["shap_values"]).mean(0),
        }
    ).sort_values("importance", ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(feature_importance["feature"][:15], feature_importance["importance"][:15])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Features by SHAP Importance")
    ax.invert_yaxis()

    plt.tight_layout()

    return feature_importance, fig


def create_shap_dependence_plots(
    shap_data: dict[str, Any],
    key_features: list,
) -> dict[str, plt.Figure]:
    """Create SHAP dependence plots for key features."""

    dependence_plots = {}

    for feature in key_features:
        if feature in shap_data["feature_names"]:
            fig, ax = plt.subplots(figsize=(10, 6))

            shap.dependence_plot(
                feature, shap_data["shap_values"], shap_data["X_test"], show=False
            )

            plt.title(f"SHAP Dependence Plot - {feature}")
            plt.tight_layout()

            dependence_plots[f"shap_dependence_{feature}"] = fig

            logger.info(f"Created SHAP dependence plot for {feature}")

    return dependence_plots


def analyze_local_interpretations(
    shap_data: dict[str, Any],
    validation_predictions: dict[str, Any],
    n_samples: int = 2,
) -> dict[str, Any]:
    """Analyze local interpretations for interesting samples."""

    y_prob = np.array(validation_predictions["y_prob"])
    y_test = np.array(validation_predictions["y_test"])

    # Find interesting samples
    high_risk_idx = np.argmax(y_prob)
    low_risk_idx = np.argmin(y_prob)

    local_interpretations = {}

    for idx, desc in [(high_risk_idx, "highest_risk"), (low_risk_idx, "lowest_risk")]:
        # Get sample information
        sample_info = {
            "index": int(idx),
            "predicted_probability": float(y_prob[idx]),
            "actual_label": "Intrusion" if y_test[idx] == 1 else "Normal",
            "description": desc,
        }

        # Get SHAP values and feature values for this sample
        sample_shap = shap_data["shap_values"][idx, :]
        sample_features = shap_data["X_test"].iloc[idx]

        # Top contributing features
        feature_contributions = pd.DataFrame(
            {
                "feature": shap_data["feature_names"],
                "shap_value": sample_shap,
                "feature_value": sample_features.values,
            }
        ).sort_values("shap_value", key=abs, ascending=False)

        # Top 5 features
        top_features = []
        for i, row in feature_contributions.head(5).iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            top_features.append(
                {
                    "feature": row["feature"],
                    "feature_value": float(row["feature_value"]),
                    "shap_value": float(row["shap_value"]),
                    "direction": direction,
                    "impact": f"{direction} risk by {abs(row['shap_value']):.3f}",
                }
            )

        sample_info["top_contributing_features"] = top_features
        local_interpretations[desc] = sample_info

        logger.info(f"Sample with {desc} (index {idx}):")
        logger.info(f"  Predicted probability: {y_prob[idx]:.3f}")
        logger.info(f"  Actual label: {sample_info['actual_label']}")
        logger.info("  Top 5 contributing features:")

        for feature_info in top_features:
            logger.info(
                f"    {feature_info['feature']} = {feature_info['feature_value']:.3f} "
                f"({feature_info['impact']})"
            )

    return local_interpretations


def create_global_feature_importance_comparison(
    feature_importance_data: pd.DataFrame,
    shap_feature_importance: pd.DataFrame,
) -> plt.Figure:
    """Create comparison between model feature importance and SHAP importance."""

    # Merge the two importance measures
    merged_importance = pd.merge(
        feature_importance_data[["feature", "importance"]].rename(
            columns={"importance": "model_importance"}
        ),
        shap_feature_importance[["feature", "importance"]].rename(
            columns={"importance": "shap_importance"}
        ),
        on="feature",
        how="outer",
    ).fillna(0)

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Model importance
    top_model = merged_importance.nlargest(15, "model_importance")
    ax1.barh(top_model["feature"], top_model["model_importance"])
    ax1.set_xlabel("Model Feature Importance")
    ax1.set_title("Top 15 Features by Model Importance")
    ax1.invert_yaxis()

    # SHAP importance
    top_shap = merged_importance.nlargest(15, "shap_importance")
    ax2.barh(top_shap["feature"], top_shap["shap_importance"])
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title("Top 15 Features by SHAP Importance")
    ax2.invert_yaxis()

    plt.tight_layout()

    return fig
