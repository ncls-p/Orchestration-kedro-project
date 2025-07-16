"""Tests for model validation pipeline nodes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from lightgbm.sklearn import LGBMClassifier

from kedro_project.pipelines.model_validation.nodes import (
    create_calibration_curve,
    create_density_chart,
    create_pr_curve,
    create_roc_curve,
    generate_probability_predictions,
    run_invariance_test,
    run_prototype_test,
)


class TestModelValidationNodes:
    """Test suite for model validation pipeline nodes."""

    @pytest.fixture
    def sample_model_and_data(self):
        """Create sample model and test data."""
        np.random.seed(42)

        # Create sample data
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.choice([0, 1], size=100))

        X_test = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            }
        )
        y_test = pd.Series(np.random.choice([0, 1], size=50))

        # Train model
        model = LGBMClassifier(
            learning_rate=0.1, num_iterations=50, verbose=-1, random_state=42
        )
        model.fit(X_train, y_train)

        return model, X_test, y_test

    @pytest.fixture
    def sample_predictions_data(self):
        """Create sample predictions data."""
        np.random.seed(42)
        return {
            "y_test": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "y_prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75],
            "test_size": 10,
            "intrusion_rate": 0.5,
        }

    def test_generate_probability_predictions(self, sample_model_and_data):
        """Test generate_probability_predictions function."""
        model, X_test, y_test = sample_model_and_data

        predictions_data = generate_probability_predictions(model, X_test, y_test)

        # Check that predictions data has correct structure
        assert isinstance(predictions_data, dict)
        assert "y_test" in predictions_data
        assert "y_prob" in predictions_data
        assert "test_size" in predictions_data
        assert "intrusion_rate" in predictions_data

        # Check that predictions are lists (JSON serializable)
        assert isinstance(predictions_data["y_test"], list)
        assert isinstance(predictions_data["y_prob"], list)
        assert isinstance(predictions_data["test_size"], int)
        assert isinstance(predictions_data["intrusion_rate"], float)

        # Check lengths match
        assert len(predictions_data["y_test"]) == len(X_test)
        assert len(predictions_data["y_prob"]) == len(X_test)
        assert predictions_data["test_size"] == len(X_test)

        # Check probability values are between 0 and 1
        assert all(0 <= p <= 1 for p in predictions_data["y_prob"])

        # Check intrusion rate calculation
        expected_rate = sum(predictions_data["y_test"]) / len(
            predictions_data["y_test"]
        )
        assert abs(predictions_data["intrusion_rate"] - expected_rate) < 1e-10

    def test_create_density_chart(self, sample_predictions_data):
        """Test create_density_chart function."""
        fig = create_density_chart(sample_predictions_data)

        # Check that figure is created
        assert isinstance(fig, plt.Figure)

        # Check that figure has axes
        assert len(fig.axes) > 0

        # Check that title is set
        ax = fig.axes[0]
        assert ax.get_title() == "Density Chart - Class Distributions"

        # Check that labels are set
        assert ax.get_xlabel() == "Predicted Probabilities"
        assert ax.get_ylabel() == "Density"

        # Close figure to prevent memory leaks
        plt.close(fig)

    def test_create_calibration_curve(self, sample_predictions_data):
        """Test create_calibration_curve function."""
        fig = create_calibration_curve(sample_predictions_data)

        # Check that figure is created
        assert isinstance(fig, plt.Figure)

        # Check that figure has axes
        assert len(fig.axes) > 0

        # Check that title is set
        ax = fig.axes[0]
        assert ax.get_title() == "Calibration Curve - Intrusion Detection Model"

        # Check that labels are set
        assert ax.get_xlabel() == "Mean predicted probability"
        assert ax.get_ylabel() == "Fraction of positives"

        # Close figure to prevent memory leaks
        plt.close(fig)

    def test_create_roc_curve(self, sample_predictions_data):
        """Test create_roc_curve function."""
        fig = create_roc_curve(sample_predictions_data)

        # Check that figure is created
        assert isinstance(fig, plt.Figure)

        # Check that figure has axes
        assert len(fig.axes) > 0

        # Check that title is set
        ax = fig.axes[0]
        assert ax.get_title() == "ROC Curve - Intrusion Detection"

        # Check that labels are set
        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"

        # Close figure to prevent memory leaks
        plt.close(fig)

    def test_create_pr_curve(self, sample_predictions_data):
        """Test create_pr_curve function."""
        fig = create_pr_curve(sample_predictions_data)

        # Check that figure is created
        assert isinstance(fig, plt.Figure)

        # Check that figure has axes
        assert len(fig.axes) > 0

        # Check that title is set
        ax = fig.axes[0]
        assert ax.get_title() == "Precision-Recall Curve - Intrusion Detection"

        # Close figure to prevent memory leaks
        plt.close(fig)

    def test_run_invariance_test(self, sample_model_and_data):
        """Test run_invariance_test function."""
        model, X_test, y_test = sample_model_and_data

        # Use an existing feature that can be perturbed
        # Make sure all values are > 1 for the test
        X_test_modified = X_test.copy()
        X_test_modified["feature1"] = np.abs(X_test_modified["feature1"]) + 2

        results = run_invariance_test(
            model, X_test_modified, test_feature="feature1"
        )

        # Check that results have correct structure
        assert isinstance(results, dict)
        assert "abs_delta_std" in results
        assert "proportion_stable" in results
        assert "test_feature" in results
        assert "test_samples" in results

        # Check that values are reasonable
        assert results["test_feature"] == "feature1"
        assert isinstance(results["abs_delta_std"], float)
        assert isinstance(results["proportion_stable"], float)
        assert isinstance(results["test_samples"], int)

        # Check that proportion is between 0 and 1
        assert 0 <= results["proportion_stable"] <= 1

    def test_run_prototype_test(self, sample_model_and_data):
        """Test run_prototype_test function."""
        model, X_test, y_test = sample_model_and_data

        results = run_prototype_test(model, X_test, n_clusters=3)

        # Check that results have correct structure
        assert isinstance(results, dict)
        assert "n_clusters" in results
        assert "prototype_predictions" in results
        assert "cluster_centers" in results

        # Check that values are reasonable
        assert results["n_clusters"] == 3
        assert len(results["prototype_predictions"]) == 3
        assert len(results["cluster_centers"]) == 3
        assert len(results["cluster_centers"][0]) == X_test.shape[1]

        # Check that predictions are probabilities
        assert all(0 <= p <= 1 for p in results["prototype_predictions"])

    def test_visualization_functions_with_edge_cases(self):
        """Test visualization functions with edge cases."""
        # Test with minimal data
        minimal_data = {
            "y_test": [0, 1],
            "y_prob": [0.3, 0.7],
            "test_size": 2,
            "intrusion_rate": 0.5,
        }

        # All visualization functions should handle minimal data
        fig1 = create_density_chart(minimal_data)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)

        fig2 = create_calibration_curve(minimal_data)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)

        fig3 = create_roc_curve(minimal_data)
        assert isinstance(fig3, plt.Figure)
        plt.close(fig3)

        fig4 = create_pr_curve(minimal_data)
        assert isinstance(fig4, plt.Figure)
        plt.close(fig4)

    def test_invariance_test_with_missing_feature(self, sample_model_and_data):
        """Test invariance test when feature is missing."""
        model, X_test, y_test = sample_model_and_data

        # Test with a feature that doesn't exist
        with pytest.raises(ValueError, match="Feature 'NonExistentFeature' not found in test data"):
            run_invariance_test(model, X_test, test_feature="NonExistentFeature")

    def test_probability_predictions_data_types(self, sample_model_and_data):
        """Test that probability predictions have correct data types."""
        model, X_test, y_test = sample_model_and_data

        predictions_data = generate_probability_predictions(model, X_test, y_test)

        # Check that all values are JSON serializable
        import json

        try:
            json.dumps(predictions_data)
        except TypeError:
            pytest.fail("Predictions data is not JSON serializable")

        # Check specific types
        assert all(isinstance(val, int) for val in predictions_data["y_test"])
        assert all(isinstance(val, float) for val in predictions_data["y_prob"])
        assert isinstance(predictions_data["test_size"], int)
        assert isinstance(predictions_data["intrusion_rate"], float)

    def test_invariance_test_edge_cases(self, sample_model_and_data):
        """Test invariance test with various edge cases."""
        model, X_test, y_test = sample_model_and_data

        # Test with feature that has all zeros (should raise ValueError)
        X_test_zeros = X_test.copy()
        X_test_zeros["feature1"] = 0

        with pytest.raises(ValueError, match="No data remains after filtering"):
            run_invariance_test(model, X_test_zeros, test_feature="feature1")

        # Test with feature that has very small values (also should raise ValueError)
        X_test_small = X_test.copy()
        X_test_small["feature1"] = 0.5

        with pytest.raises(ValueError, match="No data remains after filtering"):
            run_invariance_test(model, X_test_small, test_feature="feature1")
