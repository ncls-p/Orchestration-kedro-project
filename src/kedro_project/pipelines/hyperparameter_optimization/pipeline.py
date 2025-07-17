from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    optimize_hyperparameters,
    train_optimized_model,
)


def create_pipeline(**_kwargs) -> Pipeline:
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
        ]
    )
