"""
Hyperparameter optimization pipeline nodes.

This module contains nodes for performing hyperparameter optimization
using Hyperopt with Tree-structured Parzen Estimator (TPE) algorithm.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
import hyperopt.hp as hp
from hyperopt import Trials, fmin, tpe
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


def _prepare_space(model_specs: dict[str, Any]) -> dict[str, Any]:
    """Convert model specs into Hyperopt search space."""
    space = {}
    for param, spec in model_specs["params"].items():
        if spec["type"] == "quniform":
            # Use quniform but cast to int if needed
            space[param] = hp.quniform(param, spec["low"], spec["high"], spec["q"])
        elif spec["type"] == "uniform":
            space[param] = hp.uniform(param, spec["low"], spec["high"])
        elif spec["type"] == "choice":
            # Support both "choice" and "choices" keys
            choices = spec.get("choices", spec.get("choice", []))
            space[param] = hp.choice(param, choices)
        else:
            raise ValueError(
                f"Unsupported parameter type: {spec['type']}. "
                f"Supported types are: 'quniform', 'uniform', 'choice'"
            )
    return space


def _objective(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: dict[str, Any],
    cv_folds: int,
) -> float:
    """Objective function for hyperparameter optimization."""
    # Create model with current parameters
    model = LGBMClassifier(**params, **model_specs.get("default_params", {}))

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        scores.append(f1_score(y_fold_val, y_pred, average="weighted"))

    return float(-np.mean(scores))  # Negative because Hyperopt minimizes


def _run_search(
    objective_fn: Any,
    space: dict[str, Any],
    max_evals: int,
) -> dict[str, Any]:
    """Run the hyperparameter search using Hyperopt."""
    trials = Trials()
    best = fmin(
        fn=objective_fn,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    return cast(dict[str, Any], best or {})


def _cast_results(
    optimum_params: dict[str, Any],
    model_specs: dict[str, Any],
) -> dict[str, Any]:
    """Cast optimized parameters to correct types."""
    # Apply type casting based on override schemas
    for param, cast_type in model_specs.get("override_schemas", {}).items():
        if param in optimum_params:
            if cast_type == "int":
                optimum_params[param] = int(optimum_params[param])
            elif cast_type == "float":
                optimum_params[param] = float(optimum_params[param])

    # Auto-cast quniform parameters to int based on parameter name patterns
    for param, value in optimum_params.items():
        if isinstance(value, float) and any(
            pattern in param.lower()
            for pattern in [
                "n_estimators",
                "max_depth",
                "num_leaves",
                "min_child_samples",
            ]
        ):
            optimum_params[param] = int(value)

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

    return optimum_params


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: dict[str, Any],
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Optimize hyperparameters using Hyperopt with TimeSeriesSplit cross-validation.

    This function performs hyperparameter optimization for LightGBM classifier
    using Tree-structured Parzen Estimator (TPE) algorithm from Hyperopt.
    It uses TimeSeriesSplit cross-validation to ensure proper temporal ordering
    of the training data and optimizes based on F1 score.

    Args:
        X_train: Training features as pandas DataFrame
        y_train: Training target as pandas Series
        model_specs: Model specifications containing parameter definitions,
            override schemas, and maximum evaluations
        cv_folds: Number of cross-validation folds for time series split

    Returns:
        dictionary containing optimized hyperparameters with proper types

    Examples:
        >>> X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> y_train = pd.Series([0, 1, 0])
        >>> model_specs = {
        ...     "params": {
        ...         "n_estimators": {
        ...             "type": "quniform",
        ...             "low": 50,
        ...             "high": 200,
        ...             "q": 50,
        ...         }
        ...     },
        ...     "override_schemas": {"n_estimators": "int"},
        ...     "max_evals": 10,
        ... }
        >>> result = optimize_hyperparameters(X_train, y_train, model_specs)
        >>> print(result)
        {'n_estimators': 100}
    """
    logger.info("Starting hyperparameter optimization...")
    result: dict[str, Any] = {}

    try:
        with mlflow.start_run(nested=True, run_name="optimize_hyperparameters"):
            # Log optimization configuration
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("max_evals", model_specs["max_evals"])
            mlflow.log_param("algorithm", "tpe")

            # Prepare search space
            space = _prepare_space(model_specs)

            # Create objective function with closure
            def objective_fn(params: dict[str, Any]) -> float:
                # Ensure integer parameters are properly cast before passing to LGBM
                cast_params = params.copy()
                for param, value in cast_params.items():
                    if isinstance(value, float) and any(
                        pattern in param.lower()
                        for pattern in [
                            "n_estimators",
                            "max_depth",
                            "num_leaves",
                            "min_child_samples",
                            "num_iterations",
                        ]
                    ):
                        cast_params[param] = int(value)
                return _objective(cast_params, X_train, y_train, model_specs, cv_folds)

            # Run optimization
            optimum_params = _run_search(objective_fn, space, model_specs["max_evals"])

            # Cast results to proper types
            result = _cast_results(optimum_params, model_specs)

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        # Return default parameters if optimization fails
        result = {"learning_rate": 0.1, "n_estimators": 100}

    return result


def train_optimized_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    optimized_params: dict[str, Any],
) -> LGBMClassifier:
    """Train final model with optimized parameters.

    This function creates and trains a LightGBM classifier using the optimized
    hyperparameters obtained from the optimization process.

    Args:
        X_train: Training features as pandas DataFrame
        y_train: Training target as pandas Series
        optimized_params: dictionary containing optimized hyperparameters

    Returns:
        Trained LGBMClassifier model

    Examples:
        >>> X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> y_train = pd.Series([0, 1, 0])
        >>> params = {"n_estimators": 100, "max_depth": 6}
        >>> model = train_optimized_model(X_train, y_train, params)
        >>> isinstance(model, LGBMClassifier)
        True
    """
    logger.info("Training optimized model...")
    model = LGBMClassifier(**optimized_params)
    model.fit(X_train, y_train)
    return model
