from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    generate_feature_importance,
    optimize_hyperparameters,
    train_optimized_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=optimize_hyperparameters,
                inputs=["X_train", "y_train", "params:model_specs", "params:cv_folds"],
                outputs="optimized_hyperparameters",
                name="optimize_hyperparameters",
            ),
            node(
                func=train_optimized_model,
                inputs=["X_train", "y_train", "optimized_hyperparameters"],
                outputs="trained_model",
                name="train_optimized_model",
            ),
            node(
                func=evaluate_model,
                inputs=["trained_model", "X_test", "y_test"],
                outputs="model_performance_metrics",
                name="evaluate_model",
            ),
            node(
                func=generate_feature_importance,
                inputs=["trained_model", "params:feature_columns"],
                outputs="feature_importance_data",
                name="generate_feature_importance",
            ),
        ]
    )
