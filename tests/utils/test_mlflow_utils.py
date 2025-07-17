"""Unit tests for kedro_project.utils.mlflow_utils.

All external MLflow interactions are patched to avoid real side-effects.
"""

from unittest.mock import Mock, patch

import pytest

import kedro_project.utils.mlflow_utils as mut


# -----------------------------------------------------------------------------#
# start_mlflow_run
# -----------------------------------------------------------------------------#
@patch("kedro_project.utils.mlflow_utils.mlflow.start_run")
@patch("kedro_project.utils.mlflow_utils.mlflow.set_experiment")
def test_start_mlflow_run_success(mock_set_experiment, mock_start_run):
    """An MLflow run should be started and the returned object forwarded."""
    run_stub = Mock()
    mock_start_run.return_value = run_stub

    result = mut.start_mlflow_run("exp-name", run_name="my-run", tags={"k": "v"})

    mock_set_experiment.assert_called_once_with("exp-name")
    mock_start_run.assert_called_once_with(run_name="my-run", tags={"k": "v"})
    assert result is run_stub


@patch("kedro_project.utils.mlflow_utils.mlflow.start_run")
@patch("kedro_project.utils.mlflow_utils.mlflow.set_experiment")
def test_start_mlflow_run_error(mock_set_experiment, mock_start_run):
    """Any MLflow exception should be re-raised so callers can react."""
    mock_start_run.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        mut.start_mlflow_run("exp")


# -----------------------------------------------------------------------------#
# log_model_metrics
# -----------------------------------------------------------------------------#
@patch("kedro_project.utils.mlflow_utils.mlflow.log_metric")
def test_log_model_metrics_numeric_only(mock_log_metric):
    """Only numeric values should be forwarded to mlflow.log_metric."""
    mut.log_model_metrics({"acc": 0.9, "note": "skip-me", "loss": 0.1}, prefix="test_")

    calls = [("test_acc", 0.9), ("test_loss", 0.1)]
    assert [tuple(c.args) for c in mock_log_metric.call_args_list] == calls


@patch(
    "kedro_project.utils.mlflow_utils.mlflow.log_metric", side_effect=Exception("err")
)
def test_log_model_metrics_swallow_error(_patched):
    """Errors must be swallowed to avoid breaking the pipeline."""
    # Should not raise
    mut.log_model_metrics({"m": 1.0})


# -----------------------------------------------------------------------------#
# get_model_from_registry
# -----------------------------------------------------------------------------#
@patch("kedro_project.utils.mlflow_utils.mlflow.lightgbm.load_model")
def test_get_model_from_registry_success(mock_load):
    """Model URI must follow the expected pattern."""
    mock_model = Mock()
    mock_load.return_value = mock_model

    result = mut.get_model_from_registry("model-name", stage="Staging")

    mock_load.assert_called_once_with("models:/model-name/Staging")
    assert result is mock_model


@patch(
    "kedro_project.utils.mlflow_utils.mlflow.lightgbm.load_model",
    side_effect=ValueError,
)
def test_get_model_from_registry_error(_patched):
    """The helper should re-raise loader exceptions unchanged."""
    with pytest.raises(ValueError):
        mut.get_model_from_registry("m")


# -----------------------------------------------------------------------------#
# register_model
# -----------------------------------------------------------------------------#
@patch("kedro_project.utils.mlflow_utils.MlflowClient")
@patch(
    "kedro_project.utils.mlflow_utils.mlflow.register_model", side_effect=RuntimeError
)
def test_register_model_logs_error(mock_register, mock_client, caplog):
    """When mlflow.register_model fails, an error must be logged (not raised)."""
    caplog.set_level("ERROR")

    mut.register_model("m", "uri")

    assert any("Failed to register model" in rec.message for rec in caplog.records)
    mock_register.assert_called_once()
    # Client should never be used when register fails
    mock_client.assert_called_once()
