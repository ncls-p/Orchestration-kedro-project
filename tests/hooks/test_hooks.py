"""Unit-tests for kedro_project.hooks.MLflowIntegrationHook.

All MLflow calls are patched so no real tracking happens.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Set, Union
from unittest.mock import Mock, patch

from kedro.pipeline.node import Node

from kedro_project.hooks import MLflowIntegrationHook


class _DummyNode(Node):
    """Minimal Node subclass for testing hooks without real Kedro pipeline."""

    def __init__(
        self,
        name: str,
        func: Optional[Callable[..., Any]] = None,
        inputs: Optional[Union[str, list[str], dict[str, str]]] = None,
        outputs: Optional[Union[str, list[str], dict[str, str]]] = None,
        tags: Optional[Set[str]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        super().__init__(
            func=func or (lambda: None),
            inputs=inputs or [],
            outputs=outputs or [],
            name=name,
            tags=tags or set(),
            namespace=namespace,
        )


# -----------------------------------------------------------------------------#
# after_node_run – hyper-parameter optimisation logging
# -----------------------------------------------------------------------------#
@patch("kedro_project.hooks.mlflow.log_param")
@patch("kedro_project.hooks.mlflow.start_run")
def test_log_hyperparameter_optimisation(mock_start_run, mock_log_param):
    hook = MLflowIntegrationHook()
    node = _DummyNode("optimize_hyperparameters")

    outputs = {"optimized_hyperparameters": {"lr": 0.03, "depth": 5}}
    hook.after_node_run(node, Mock(), {}, outputs)

    mock_start_run.assert_called_once_with(
        nested=True, run_name="hyperparameter_optimization"
    )
    # Two parameters must be forwarded
    assert mock_log_param.call_count == 2


# -----------------------------------------------------------------------------#
# after_node_run – model training logging happy path
# -----------------------------------------------------------------------------#
@patch("kedro_project.hooks.mlflow.lightgbm.log_model")
@patch("kedro_project.hooks.mlflow.start_run")
def test_log_model_training_success(mock_start_run, mock_log_model):
    hook = MLflowIntegrationHook()
    node = _DummyNode("train_optimized_model")

    inputs = {"optimized_hyperparameters": {"p": 1}}
    outputs = {"trained_model": Mock()}
    hook.after_node_run(node, Mock(), inputs, outputs)

    mock_start_run.assert_called_once_with(nested=True, run_name="model_training")
    mock_log_model.assert_called_once()


# -----------------------------------------------------------------------------#
# after_node_run – model training logging internal failure
# -----------------------------------------------------------------------------#
@patch("kedro_project.hooks.logger")
@patch(
    "kedro_project.hooks.mlflow.lightgbm.log_model", side_effect=RuntimeError("fail")
)
@patch("kedro_project.hooks.mlflow.start_run")
def test_log_model_training_handles_error(mock_start_run, _patched, mock_logger):
    hook = MLflowIntegrationHook()
    node = _DummyNode("train_optimized_model")

    hook.after_node_run(node, Mock(), {}, {"trained_model": Mock()})

    # Error must be logged, not raised
    mock_logger.error.assert_called_once()
    error_call_args = mock_logger.error.call_args[0]
    assert len(error_call_args) == 1
    assert "Failed to log model" in str(error_call_args[0])


# -----------------------------------------------------------------------------#
# after_node_run – model evaluation metrics logging
# -----------------------------------------------------------------------------#
@patch("kedro_project.hooks.mlflow.log_metric")
@patch("kedro_project.hooks.mlflow.start_run")
def test_log_model_evaluation(mock_start_run, mock_log_metric):
    hook = MLflowIntegrationHook()
    node = _DummyNode("evaluate_model")

    outputs = {
        "model_performance_metrics": {"f1_score": 0.9, "precision": 0.8, "recall": 0.7}
    }
    hook.after_node_run(node, Mock(), {}, outputs)

    mock_start_run.assert_called_once_with(nested=True, run_name="model_evaluation")
    # Three metrics expected
    assert mock_log_metric.call_count == 3


# -----------------------------------------------------------------------------#
# before_node_run – ensures no exceptions are emitted
# -----------------------------------------------------------------------------#
def test_before_node_run_noop():
    hook = MLflowIntegrationHook()
    # Any unknown node should not raise
    hook.before_node_run(_DummyNode("some_node"), Mock(), {}, False)
