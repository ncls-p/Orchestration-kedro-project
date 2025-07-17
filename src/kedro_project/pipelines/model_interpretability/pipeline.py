from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    analyze_local_interpretations,
    compute_shap_values,
    create_global_feature_importance_comparison,
    create_shap_dependence_plots,
    create_shap_feature_importance,
    create_shap_summary_plot,
)


def create_pipeline(**_kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=compute_shap_values,
                inputs=["trained_model", "X_test"],
                outputs="shap_data",
                name="compute_shap_values",
            ),
            node(
                func=create_shap_summary_plot,
                inputs="shap_data",
                outputs="shap_summary_plot",
                name="create_shap_summary_plot",
            ),
            node(
                func=create_shap_feature_importance,
                inputs="shap_data",
                outputs=["shap_feature_importance", "shap_feature_importance_plot"],
                name="create_shap_feature_importance",
            ),
            node(
                func=create_shap_dependence_plots,
                inputs=["shap_data", "params:key_features_for_dependence"],
                outputs="shap_dependence_plots",
                name="create_shap_dependence_plots",
            ),
            node(
                func=analyze_local_interpretations,
                inputs=[
                    "shap_data",
                    "validation_predictions",
                    "params:n_local_samples",
                ],
                outputs="local_interpretations",
                name="analyze_local_interpretations",
            ),
            node(
                func=create_global_feature_importance_comparison,
                inputs=["feature_importance_data", "shap_feature_importance"],
                outputs="feature_importance_comparison_plot",
                name="create_global_feature_importance_comparison",
            ),
        ]
    )
