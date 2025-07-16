"""Tests for hyperparameter optimization pipeline nodes."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lightgbm.sklearn import LGBMClassifier

from kedro_project.pipelines.hyperparameter_optimization.nodes import (
    _convert_params_to_hyperopt,
    evaluate_model,
    generate_feature_importance,
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
    def sample_test_data(self):
        """Create sample test data for testing."""
        np.random.seed(42)
        X_test = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            }
        )
        y_test = pd.Series(np.random.choice([0, 1], size=50))
        return X_test, y_test

    @pytest.fixture
    def sample_model_specs(self):
        """Create sample model specifications for testing."""
        return {
            "name": "LightGBM",
            "max_evals": 5,  # Small number for testing
            "params": {
                "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.3},
                "num_iterations": {"type": "quniform", "low": 50, "high": 200, "q": 50},
                "max_depth": {"type": "quniform", "low": 3, "high": 8, "q": 1},
                "reg_alpha": {"type": "choice", "options": [0, 0.1, 0.5]},
            },
            "override_schemas": {"num_iterations": "int", "max_depth": "int"},
        }

    def test_convert_params_to_hyperopt(self, sample_model_specs):
        """Test _convert_params_to_hyperopt function."""
        hyperopt_params = _convert_params_to_hyperopt(sample_model_specs["params"])

        # Check that all parameters are converted
        assert "learning_rate" in hyperopt_params
        assert "num_iterations" in hyperopt_params
        assert "max_depth" in hyperopt_params
        assert "reg_alpha" in hyperopt_params

        # Check that hyperopt objects are created
        assert hasattr(hyperopt_params["learning_rate"], "name")
        assert hasattr(hyperopt_params["num_iterations"], "name")
        assert hasattr(hyperopt_params["max_depth"], "name")
        assert hasattr(hyperopt_params["reg_alpha"], "name")

    @patch("kedro_project.pipelines.hyperparameter_optimization.nodes.fmin")
    def test_optimize_hyperparameters(
        self, mock_fmin, sample_train_data, sample_model_specs
    ):
        """Test optimize_hyperparameters function."""
        X_train, y_train = sample_train_data

        # Mock fmin to return sample parameters
        mock_fmin.return_value = {
            "learning_rate": 0.1,
            "num_iterations": 100.0,
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
        assert "num_iterations" in result
        assert "max_depth" in result
        assert "reg_alpha" in result

        # Check that integer casting works
        assert isinstance(result["num_iterations"], int)
        assert isinstance(result["max_depth"], int)
        assert isinstance(result["learning_rate"], float)

    def test_train_optimized_model(self, sample_train_data):
        """Test train_optimized_model function."""
        X_train, y_train = sample_train_data

        optimized_params = {
            "learning_rate": 0.1,
            "num_iterations": 100,
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
        assert model.num_iterations == 100
        assert model.max_depth == 5
        assert model.reg_alpha == 0.1

    def test_evaluate_model(self, sample_train_data, sample_test_data):
        """Test evaluate_model function."""
        X_train, y_train = sample_train_data
        X_test, y_test = sample_test_data

        # Train a simple model
        model = LGBMClassifier(
            learning_rate=0.1, num_iterations=50, verbose=-1, random_state=42
        )
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "classification_report" in metrics

        # Check that metric values are reasonable
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert isinstance(metrics["classification_report"], str)

    def test_generate_feature_importance(self, sample_train_data):
        """Test generate_feature_importance function."""
        X_train, y_train = sample_train_data

        # Train a simple model
        model = LGBMClassifier(
            learning_rate=0.1, num_iterations=50, verbose=-1, random_state=42
        )
        model.fit(X_train, y_train)

        feature_columns = ["feature1", "feature2", "feature3"]
        importance_df = generate_feature_importance(model, feature_columns)

        # Check that importance DataFrame is correct
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(feature_columns)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns

        # Check that features are sorted by importance
        assert importance_df["importance"].is_monotonic_decreasing

        # Check that all features are present
        assert set(importance_df["feature"].values) == set(feature_columns)

    def test_optimize_hyperparameters_with_actual_optimization(self, sample_train_data):
        """Test optimize_hyperparameters with actual optimization (integration test)."""
        X_train, y_train = sample_train_data

        # Use a very simple model spec for faster testing
        simple_model_specs = {
            "name": "LightGBM",
            "max_evals": 3,  # Very small number for testing
            "params": {
                "learning_rate": {"type": "uniform", "low": 0.05, "high": 0.2},
                "num_iterations": {"type": "quniform", "low": 50, "high": 100, "q": 50},
            },
            "override_schemas": {"num_iterations": "int"},
        }

        result = optimize_hyperparameters(
            X_train, y_train, simple_model_specs, cv_folds=2
        )

        # Check that optimization returns reasonable parameters
        assert isinstance(result, dict)
        assert "learning_rate" in result
        assert "num_iterations" in result

        # Check parameter ranges
        assert 0.05 <= result["learning_rate"] <= 0.2
        assert result["num_iterations"] in [50, 100]
        assert isinstance(result["num_iterations"], int)

    def test_model_with_empty_params(self, sample_train_data):
        """Test train_optimized_model with empty parameters."""
        X_train, y_train = sample_train_data

        model = train_optimized_model(X_train, y_train, {})

        # Check that model is trained with default parameters
        assert isinstance(model, LGBMClassifier)
        assert hasattr(model, "predict")

    def test_evaluate_model_edge_cases(self, sample_train_data):
        """Test evaluate_model with edge cases."""
        X_train, y_train = sample_train_data

        # Create a model that always predicts the same class
        model = LGBMClassifier(
            learning_rate=0.1, num_iterations=10, verbose=-1, random_state=42
        )
        model.fit(X_train, y_train)

        # Test with small dataset
        X_small = pd.DataFrame(
            {"feature1": [1, 2], "feature2": [3, 4], "feature3": [5, 6]}
        )
        y_small = pd.Series([0, 1])

        metrics = evaluate_model(model, X_small, y_small)

        # Should still return valid metrics
        assert isinstance(metrics, dict)
        assert "f1_score" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_hyperopt_parameter_types(self):
        """Test that hyperopt parameters are created with correct types."""
        params_config = {
            "uniform_param": {"type": "uniform", "low": 0.1, "high": 1.0},
            "quniform_param": {"type": "quniform", "low": 10, "high": 100, "q": 10},
            "choice_param": {"type": "choice", "options": ["a", "b", "c"]},
        }

        hyperopt_params = _convert_params_to_hyperopt(params_config)

        # Check that all parameters are converted
        assert len(hyperopt_params) == 3
        assert "uniform_param" in hyperopt_params
        assert "quniform_param" in hyperopt_params
        assert "choice_param" in hyperopt_params

        # Check that hyperopt objects have correct attributes
        assert hasattr(hyperopt_params["uniform_param"], "name")
        assert hasattr(hyperopt_params["quniform_param"], "name")
        assert hasattr(hyperopt_params["choice_param"], "name")
