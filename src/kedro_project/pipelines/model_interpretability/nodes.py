import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm.sklearn import LGBMClassifier
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set seaborn theme
sns.set_theme()

__all__ = [
    "compute_shap_values",
    "create_shap_summary_plot",
    "create_shap_feature_importance",
    "create_shap_dependence_plots",
    "analyze_local_interpretations",
    "create_global_feature_importance_comparison",
    "extract_feature_importance",
]


def extract_feature_importance(
    model: LGBMClassifier,
) -> pd.DataFrame:
    """
    Extract feature importance from trained LightGBM model.

    Args:
        model: Trained LightGBM classifier model

    Returns:
        DataFrame with feature names and their importance scores

    Examples:
        >>> model = LGBMClassifier()
        >>> importance_df = extract_feature_importance(model)
        >>> print(importance_df.columns.tolist())
        ['feature', 'importance']
    """
    logger.info("Extracting feature importance from model...")

    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = model.feature_name_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    logger.info(f"Extracted importance for {len(importance_df)} features")

    return importance_df


def compute_shap_values(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute SHAP values for model interpretability.

    This function creates a SHAP explainer for the trained model and computes
    SHAP values for the test dataset, handling binary classification output
    by extracting positive class values.

    Args:
        model: Trained LightGBM classifier model
        X_test: Test features DataFrame to compute SHAP values for

    Returns:
        Dictionary containing:
            - shap_values: SHAP values array for positive class
            - feature_names: List of feature names
            - X_test: Original test features DataFrame

    Examples:
        >>> model = LGBMClassifier()
        >>> X_test = pd.DataFrame([[1, 2], [3, 4]], columns=["feat1", "feat2"])
        >>> shap_data = compute_shap_values(model, X_test)
        >>> print(shap_data.keys())
        dict_keys(['shap_values', 'feature_names', 'X_test'])
    """
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


def create_shap_summary_plot(shap_data: dict[str, Any]) -> Figure:
    """
    Create SHAP summary plot.

    Generates a SHAP summary plot that shows feature importance and the
    impact of each feature on model predictions for intrusion detection.

    Args:
        shap_data: Dictionary containing SHAP values, feature names, and test data

    Returns:
        matplotlib Figure containing the SHAP summary plot

    Examples:
        >>> shap_data = {
        ...     "shap_values": np.array([[0.1, -0.2], [0.3, 0.4]]),
        ...     "X_test": pd.DataFrame([[1, 2], [3, 4]]),
        ... }
        >>> fig = create_shap_summary_plot(shap_data)
        >>> fig.savefig("shap_summary.png")
    """
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
) -> tuple[pd.DataFrame, Figure]:
    """
    Create SHAP-based feature importance plot.

    Calculates feature importance based on mean absolute SHAP values
    and creates a horizontal bar plot showing the top 15 most important features.

    Args:
        shap_data: Dictionary containing SHAP values and feature names

    Returns:
        Tuple containing:
            - pd.DataFrame: Feature importance data sorted by importance
            - plt.Figure: Feature importance plot

    Examples:
        >>> shap_data = {
        ...     "shap_values": np.array([[0.1, -0.2], [0.3, 0.4]]),
        ...     "feature_names": ["feature1", "feature2"],
        ... }
        >>> importance_df, fig = create_shap_feature_importance(shap_data)
        >>> print(importance_df.columns.tolist())
        ['feature', 'importance']
    """
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
    key_features: list[str],
) -> dict[str, Figure]:
    """
    Create SHAP dependence plots for key features.

    Generates SHAP dependence plots for specified key features, showing
    how each feature's value affects the model's output through SHAP values.

    Args:
        shap_data: Dictionary containing SHAP values, feature names, and test data
        key_features: List of feature names to create dependence plots for

    Returns:
        Dictionary mapping plot names to matplotlib Figure objects

    Examples:
        >>> shap_data = {
        ...     "shap_values": np.array([[0.1, -0.2], [0.3, 0.4]]),
        ...     "feature_names": ["feature1", "feature2"],
        ...     "X_test": pd.DataFrame(
        ...         [[1, 2], [3, 4]], columns=["feature1", "feature2"]
        ...     ),
        ... }
        >>> key_features = ["feature1"]
        >>> plots = create_shap_dependence_plots(shap_data, key_features)
        >>> "shap_dependence_feature1" in plots
        True
    """
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


def _find_interesting_samples(
    validation_predictions: dict[str, Any], n_samples: int
) -> list[tuple[int, str]]:
    """
    Find interesting samples for local interpretation analysis.

    Args:
        validation_predictions: Dictionary containing y_prob and y_test arrays
        n_samples: Number of samples to find (currently unused, always returns 2)

    Returns:
        List of tuples containing (index, description) for interesting samples
    """
    y_prob = np.array(validation_predictions["y_prob"])

    # Find interesting samples
    high_risk_idx = int(np.argmax(y_prob))
    low_risk_idx = int(np.argmin(y_prob))

    return [(high_risk_idx, "highest_risk"), (low_risk_idx, "lowest_risk")]


def _extract_sample_info(
    idx: int, desc: str, y_prob: np.ndarray, y_test: np.ndarray
) -> dict[str, Any]:
    """
    Extract basic information for a sample.

    Args:
        idx: Sample index
        desc: Sample description
        y_prob: Prediction probabilities array
        y_test: True labels array

    Returns:
        Dictionary containing sample information
    """
    return {
        "index": int(idx),
        "predicted_probability": float(y_prob[idx]),
        "actual_label": "Intrusion" if y_test[idx] == 1 else "Normal",
        "description": desc,
    }


def _create_feature_contributions_df(
    shap_data: dict[str, Any], idx: int
) -> pd.DataFrame:
    """
    Create DataFrame with feature contributions for a sample.

    Args:
        shap_data: Dictionary containing SHAP values and feature information
        idx: Sample index

    Returns:
        DataFrame with feature contributions sorted by absolute SHAP value
    """
    sample_shap = shap_data["shap_values"][idx, :]
    sample_features = shap_data["X_test"].iloc[idx]

    feature_contributions = pd.DataFrame(
        {
            "feature": shap_data["feature_names"],
            "shap_value": sample_shap,
            "feature_value": sample_features.values,
        }
    ).sort_values("shap_value", key=abs, ascending=False)

    return feature_contributions


def _format_feature_impact(row: pd.Series) -> dict[str, Any]:
    """
    Format feature impact information for a single feature.

    Args:
        row: Pandas Series containing feature, shap_value, and feature_value

    Returns:
        Dictionary with formatted feature impact information
    """
    direction = "increases" if row["shap_value"] > 0 else "decreases"
    return {
        "feature": row["feature"],
        "feature_value": float(row["feature_value"]),
        "shap_value": float(row["shap_value"]),
        "direction": direction,
        "impact": f"{direction} risk by {abs(row['shap_value']):.3f}",
    }


def _compute_feature_contributions(
    shap_data: dict[str, Any], idx: int
) -> list[dict[str, Any]]:
    """
    Compute top contributing features for a sample.

    Args:
        shap_data: Dictionary containing SHAP values and feature information
        idx: Sample index

    Returns:
        List of dictionaries containing feature contribution information
    """
    feature_contributions = _create_feature_contributions_df(shap_data, idx)

    # Top 5 features
    top_features = []
    for i, row in feature_contributions.head(5).iterrows():
        top_features.append(_format_feature_impact(row))

    return top_features


def _log_sample_analysis(
    idx: int,
    desc: str,
    y_prob: np.ndarray,
    sample_info: dict[str, Any],
    top_features: list[dict[str, Any]],
) -> None:
    """
    Log sample analysis information.

    Args:
        idx: Sample index
        desc: Sample description
        y_prob: Prediction probabilities array
        sample_info: Sample information dictionary
        top_features: List of top contributing features
    """
    logger.info(f"Sample with {desc} (index {idx}):")
    logger.info(f"  Predicted probability: {y_prob[idx]:.3f}")
    logger.info(f"  Actual label: {sample_info['actual_label']}")
    logger.info("  Top 5 contributing features:")

    for feature_info in top_features:
        logger.info(
            f"    {feature_info['feature']} = {feature_info['feature_value']:.3f} "
            f"({feature_info['impact']})"
        )


def analyze_local_interpretations(
    shap_data: dict[str, Any],
    validation_predictions: dict[str, Any],
    n_samples: int = 2,
) -> dict[str, Any]:
    """
    Analyze local interpretations for interesting samples.

    This function identifies samples with highest and lowest predicted probabilities
    and analyzes their SHAP values to understand which features contribute most
    to the model's predictions for these specific instances.

    Args:
        shap_data: Dictionary containing SHAP values, feature names, and test data
        validation_predictions: Dictionary containing y_prob and y_test arrays
        n_samples: Number of samples to analyze (currently unused, always analyzes 2)

    Returns:
        Dictionary containing local interpretation analysis for interesting samples

    Examples:
        >>> shap_data = {
        ...     "shap_values": np.array([[0.1, -0.2], [0.3, 0.4]]),
        ...     "feature_names": ["feature1", "feature2"],
        ...     "X_test": pd.DataFrame(
        ...         [[1, 2], [3, 4]], columns=["feature1", "feature2"]
        ...     ),
        ... }
        >>> predictions = {"y_prob": [0.1, 0.9], "y_test": [0, 1]}
        >>> results = analyze_local_interpretations(shap_data, predictions)
        >>> "highest_risk" in results and "lowest_risk" in results
        True
    """
    y_prob = np.array(validation_predictions["y_prob"])
    y_test = np.array(validation_predictions["y_test"])

    # Find interesting samples
    interesting_samples = _find_interesting_samples(validation_predictions, n_samples)

    local_interpretations = {}

    for idx, desc in interesting_samples:
        # Get sample information
        sample_info = _extract_sample_info(idx, desc, y_prob, y_test)

        # Compute feature contributions
        top_features = _compute_feature_contributions(shap_data, idx)

        sample_info["top_contributing_features"] = top_features
        local_interpretations[desc] = sample_info

        # Log analysis
        _log_sample_analysis(idx, desc, y_prob, sample_info, top_features)

    return local_interpretations


def _merge_importance_measures(
    feature_importance_data: pd.DataFrame,
    shap_feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge model and SHAP importance measures into a single DataFrame.

    Args:
        feature_importance_data: DataFrame with feature and importance columns
        shap_feature_importance: DataFrame with SHAP feature and importance columns

    Returns:
        Merged DataFrame with model_importance and shap_importance columns
    """
    model_importance_df = feature_importance_data[["feature", "importance"]].copy()
    model_importance_df.columns = ["feature", "model_importance"]

    shap_importance_df = shap_feature_importance[["feature", "importance"]].copy()
    shap_importance_df.columns = ["feature", "shap_importance"]

    merged_importance = pd.merge(
        model_importance_df,
        shap_importance_df,
        on="feature",
        how="outer",
    ).fillna(0)

    return merged_importance


def _create_comparison_subplots(merged_importance: pd.DataFrame) -> Figure:
    """
    Create side-by-side bar charts comparing model and SHAP importance.

    Args:
        merged_importance: DataFrame with model_importance and shap_importance columns

    Returns:
        Figure with comparison subplots
    """
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


def create_global_feature_importance_comparison(
    feature_importance_data: pd.DataFrame,
    shap_feature_importance: pd.DataFrame,
) -> Figure:
    """
    Create comparison between model feature importance and SHAP importance.

    This function combines model-based feature importance with SHAP values to create
    a side-by-side comparison visualization showing the top 15 features according
    to each importance measure.

    Args:
        feature_importance_data: DataFrame containing feature names and model importance scores
        shap_feature_importance: DataFrame containing feature names and SHAP importance scores

    Returns:
        Matplotlib Figure with side-by-side bar charts comparing importance measures

    Examples:
        >>> model_importance = pd.DataFrame(
        ...     {"feature": ["feature1", "feature2"], "importance": [0.8, 0.6]}
        ... )
        >>> shap_importance = pd.DataFrame(
        ...     {"feature": ["feature1", "feature2"], "importance": [0.7, 0.5]}
        ... )
        >>> fig = create_global_feature_importance_comparison(
        ...     model_importance, shap_importance
        ... )
        >>> fig.savefig("importance_comparison.png")
    """
    merged_importance = _merge_importance_measures(
        feature_importance_data, shap_feature_importance
    )
    return _create_comparison_subplots(merged_importance)
