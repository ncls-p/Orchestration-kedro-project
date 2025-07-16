import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from hyperopt import fmin, hp, tpe
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def _convert_params_to_hyperopt(params_config: dict[str, Any]) -> dict[str, Any]:
    """Convert parameter configuration to hyperopt format."""
    hyperopt_params = {}

    for param_name, param_config in params_config.items():
        if param_config["type"] == "uniform":
            hyperopt_params[param_name] = hp.uniform(
                param_name, param_config["low"], param_config["high"]
            )
        elif param_config["type"] == "quniform":
            hyperopt_params[param_name] = hp.quniform(
                param_name, param_config["low"], param_config["high"], param_config["q"]
            )
        elif param_config["type"] == "choice":
            hyperopt_params[param_name] = hp.choice(param_name, param_config["options"])

    return hyperopt_params


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: dict[str, Any],
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Optimize hyperparameters using Hyperopt with TimeSeriesSplit cross-validation."""

    # Convert parameter configuration to hyperopt format
    hyperopt_params = _convert_params_to_hyperopt(model_specs["params"])

    def objective(params):
        # Cast parameters to correct types
        for param in set(list(model_specs["override_schemas"].keys())).intersection(
            set(params.keys())
        ):
            cast_type = model_specs["override_schemas"][param]
            if cast_type == "int":
                params[param] = int(params[param])
            elif cast_type == "float":
                params[param] = float(params[param])

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores_test = []

        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            # Train model
            model = LGBMClassifier(
                **params, objective="binary", verbose=-1, random_state=42
            )
            model.fit(X_fold_train, y_fold_train)

            # Evaluate on validation set
            y_pred = model.predict(X_fold_val)
            scores_test.append(f1_score(y_fold_val, y_pred))

        return -np.mean(scores_test)  # Minimize negative F1

    logger.info("Starting hyperparameter optimization...")

    # Log optimization configuration
    mlflow.log_param("cv_folds", cv_folds)
    mlflow.log_param("max_evals", model_specs["max_evals"])
    mlflow.log_param("algorithm", "tpe")

    optimum_params = fmin(
        fn=objective,
        space=hyperopt_params,
        algo=tpe.suggest,
        max_evals=model_specs["max_evals"],
        rstate=np.random.default_rng(42),
    )

    # Cast optimized parameters
    if optimum_params is not None:
        for param in model_specs["override_schemas"]:
            if param in optimum_params:
                cast_type = model_specs["override_schemas"][param]
                if cast_type == "int":
                    optimum_params[param] = int(optimum_params[param])
                elif cast_type == "float":
                    optimum_params[param] = float(optimum_params[param])

        # Convert all numpy types to Python types for JSON serialization
        for key, value in optimum_params.items():
            if hasattr(value, "item"):  # numpy scalars
                optimum_params[key] = value.item()
            elif isinstance(value, np.integer):
                optimum_params[key] = int(value)
            elif isinstance(value, np.floating):
                optimum_params[key] = float(value)

        logger.info("Hyperparameter optimization completed successfully")
        for key, value in optimum_params.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.warning("Optimization failed, using default parameters")
        optimum_params = {}

    return optimum_params


def train_optimized_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optimized_params: dict[str, Any],
) -> LGBMClassifier:
    """Train final model with optimized parameters."""

    model = LGBMClassifier(
        **optimized_params, objective="binary", verbose=-1, random_state=42
    )
    model.fit(X_train, y_train)

    logger.info("Model training completed with optimized parameters")

    return model


def evaluate_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate model performance on test set."""

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    performance_metrics = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Normal", "Intrusion"]
        ),
    }

    logger.info("=== Model Performance ===")
    logger.info(f"F1 Score: {f1 * 100:.1f}%")
    logger.info(f"Precision: {precision * 100:.1f}%")
    logger.info(f"Recall: {recall * 100:.1f}%")
    logger.info(
        f"Classification Report:\n{performance_metrics['classification_report']}"
    )

    return performance_metrics


def generate_feature_importance(
    model: LGBMClassifier,
    feature_columns: list,
) -> pd.DataFrame:
    """Generate feature importance DataFrame."""

    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return feature_importance
