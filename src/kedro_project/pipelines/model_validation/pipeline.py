from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_calibration_curve,
    create_density_chart,
    create_pr_curve,
    create_roc_curve,
    generate_probability_predictions,
    run_invariance_test,
    run_prototype_test,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_probability_predictions,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="validation_predictions",
                name="generate_probability_predictions",
            ),
            node(
                func=create_density_chart,
                inputs="validation_predictions",
                outputs="density_chart",
                name="create_density_chart",
            ),
            node(
                func=create_calibration_curve,
                inputs="validation_predictions",
                outputs="calibration_curve",
                name="create_calibration_curve",
            ),
            node(
                func=create_roc_curve,
                inputs="validation_predictions",
                outputs="roc_curve",
                name="create_roc_curve",
            ),
            node(
                func=create_pr_curve,
                inputs="validation_predictions",
                outputs="pr_curve",
                name="create_pr_curve",
            ),
            node(
                func=run_invariance_test,
                inputs=["trained_model", "X_test", "params:invariance_test_feature"],
                outputs="invariance_test_results",
                name="run_invariance_test",
            ),
            node(
                func=run_prototype_test,
                inputs=["trained_model", "X_test", "params:prototype_n_clusters"],
                outputs="prototype_test_results",
                name="run_prototype_test",
            ),
        ]
    )
