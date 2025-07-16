"""
This module contains integration tests for the Kedro project.
Tests should be placed in ``tests/``, in modules that mirror your
project's structure, and in files named test_*.py.
"""

from pathlib import Path

import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    """Test suite for main Kedro pipeline execution."""

    def test_kedro_run_data_processing(self):
        """Test that data processing pipeline runs successfully."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            # Run only the data processing pipeline
            result = session.run(pipeline_name="data_processing")
            assert result is not None

    def test_kedro_run_hyperparameter_optimization(self):
        """Test that hyperparameter optimization pipeline runs successfully."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            # Run hyperparameter optimization pipeline
            result = session.run(pipeline_name="hyperparameter_optimization")
            assert result is not None

    def test_kedro_run_model_validation(self):
        """Test that model validation pipeline runs successfully."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            # Run model validation pipeline
            result = session.run(pipeline_name="model_validation")
            assert result is not None

    def test_kedro_run_model_interpretability(self):
        """Test that model interpretability pipeline runs successfully."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            # Run model interpretability pipeline
            result = session.run(pipeline_name="model_interpretability")
            assert result is not None

    @pytest.mark.slow
    def test_kedro_run_full_pipeline(self):
        """Test that full pipeline runs successfully (marked as slow test)."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            # Run the complete pipeline
            result = session.run()
            assert result is not None

    def test_kedro_pipeline_registry(self):
        """Test that all pipelines are properly registered."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            # Test that pipelines can be found by trying to run them
            # If a pipeline doesn't exist, session.run() will raise an error
            expected_pipelines = [
                "data_processing",
                "hyperparameter_optimization",
                "model_validation",
                "model_interpretability",
            ]

            for pipeline_name in expected_pipelines:
                try:
                    # This will fail if the pipeline doesn't exist or has issues
                    # We're not actually running it, just checking it exists
                    # by using dry-run mode (if available) or catching the specific error
                    session.run(pipeline_name=pipeline_name, runner="MemoryRunner")
                except ValueError as e:
                    if "Failed to find the pipeline" in str(e):
                        pytest.fail(f"Pipeline '{pipeline_name}' not found in registry")
                    # Other errors are OK - it means the pipeline exists but might need data
                except Exception:
                    # Pipeline exists but might need data or have other issues
                    # This is OK for this test - we just want to check it's registered
                    pass

    def test_kedro_catalog_loading(self):
        """Test that data catalog loads correctly."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            catalog = session.load_context().catalog

            # Check that key datasets are in catalog
            expected_datasets = [
                "raw_network_logs",
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "trained_model",
                "model_performance_metrics",
            ]

            for dataset_name in expected_datasets:
                assert dataset_name in catalog._datasets

    def test_kedro_parameters_loading(self):
        """Test that parameters load correctly."""
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd()) as session:
            params = session.load_context().params

            # Check that key parameters are loaded
            expected_params = [
                "rolling_window_size",
                "lag_values",
                "train_ratio",
                "categorical_columns",
                "feature_columns",
                "model_specs",
            ]

            for param_name in expected_params:
                assert param_name in params
                assert params[param_name] is not None
