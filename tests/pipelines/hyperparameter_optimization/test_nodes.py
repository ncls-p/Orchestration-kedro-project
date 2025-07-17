"""Tests for hyperparameter optimization pipeline nodes."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lightgbm.sklearn import LGBMClassifier

from kedro_project.pipelines.hyperparameter_optimization.nodes import (
    optimize_hyperparameters,
    train_optimized_model,
)


class TestHyperparameterOptimizationNodes:
    """Test suite for hyperparameter optimization pipeline nodes."""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.choice([0, 1], size=100))
        return X_train, y_train

    @pytest.fixture
    def sample_model_specs(self):
        """Create sample model specifications for testing."""
        return {
            "name": "LightGBM",
            "max_evals": 5,  # Small number for testing
            "params": {
                "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.3},
                "n_estimators": {"type": "quniform", "low": 50, "high": 200, "q": 50},
                "max_depth": {"type": "quniform", "low": 3, "high": 8, "q": 1},
                "reg_alpha": {"type": "choice", "options": [0, 0.1, 0.5]},
            },
            "override_schemas": {"n_estimators": "int", "max_depth": "int"},
        }

    @patch("kedro_project.pipelines.hyperparameter_optimization.nodes.fmin")
    def test_optimize_hyperparameters(
        self, mock_fmin, sample_train_data, sample_model_specs
    ):
        """Test optimize_hyperparameters function."""
        X_train, y_train = sample_train_data

        # Mock fmin to return sample parameters
        mock_fmin.return_value = {
            "learning_rate": 0.1,
            "n_estimators": 100.0,
            "max_depth": 5.0,
            "reg_alpha": 0,
        }

        result = optimize_hyperparameters(
            X_train, y_train, sample_model_specs, cv_folds=2
        )

        # Check that fmin was called
        mock_fmin.assert_called_once()

        # Check that parameters are returned and properly cast
        assert isinstance(result, dict)
        assert "learning_rate" in result
        assert "n_estimators" in result
        assert "max_depth" in result
        assert "reg_alpha" in result

        # Check that integer casting works
        assert isinstance(result["n_estimators"], int)
        assert isinstance(result["max_depth"], int)
        assert isinstance(result["learning_rate"], float)

    def test_train_optimized_model(self, sample_train_data):
        """Test train_optimized_model function."""
        X_train, y_train = sample_train_data

        optimized_params = {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 5,
            "reg_alpha": 0.1,
        }

        model = train_optimized_model(X_train, y_train, optimized_params)

        # Check that model is trained
        assert isinstance(model, LGBMClassifier)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        # Check that model parameters are set correctly
        assert model.learning_rate == 0.1
        assert model.n_estimators == 100
        assert model.max_depth == 5
        assert model.reg_alpha == 0.1

    def test_optimize_hyperparameters_with_actual_optimization(self, sample_train_data):
        """Test optimize_hyperparameters with actual optimization (integration test)."""
        X_train, y_train = sample_train_data

        # Use a very simple model spec for faster testing
        simple_model_specs = {
            "name": "LightGBM",
            "max_evals": 3,  # Very small number for testing
            "params": {
                "learning_rate": {"type": "uniform", "low": 0.05, "high": 0.2},
                "n_estimators": {"type": "quniform", "low": 50, "high": 100, "q": 50},
            },
            "override_schemas": {"n_estimators": "int"},
        }

        result = optimize_hyperparameters(
            X_train, y_train, simple_model_specs, cv_folds=2
        )

        # Check that optimization returns reasonable parameters
        assert isinstance(result, dict)
        assert "learning_rate" in result
        assert "n_estimators" in result

        # Check parameter ranges
        assert 0.05 <= result["learning_rate"] <= 0.2
        assert result["n_estimators"] in [50, 100]
        assert isinstance(result["n_estimators"], int)

    def test_train_optimized_model_with_empty_params(self, sample_train_data):
        """Test train_optimized_model with empty parameters."""
        X_train, y_train = sample_train_data

        model = train_optimized_model(X_train, y_train, {})

        # Check that model is trained with default parameters
        assert isinstance(model, LGBMClassifier)
        assert hasattr(model, "predict")
