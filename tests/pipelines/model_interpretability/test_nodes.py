"""Tests for model interpretability pipeline nodes."""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from lightgbm.sklearn import LGBMClassifier

from kedro_project.pipelines.model_interpretability.nodes import (
    analyze_local_interpretations,
    compute_shap_values,
    create_global_feature_importance_comparison,
    create_shap_dependence_plots,
    create_shap_feature_importance,
    create_shap_summary_plot,
)


class TestModelInterpretabilityNodes:
    """Test suite for model interpretability pipeline nodes."""

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

        # Train model
        model = LGBMClassifier(
            learning_rate=0.1, num_iterations=50, verbose=-1, random_state=42
        )
        model.fit(X_train, y_train)

        return model, X_test

    @pytest.fixture
    def sample_shap_data(self):
        """Create sample SHAP data."""
        np.random.seed(42)

        X_test = pd.DataFrame(
            {
                "feature1": np.random.randn(20),
                "feature2": np.random.randn(20),
                "feature3": np.random.randn(20),
            }
        )

        # Create mock SHAP values
        shap_values = np.random.randn(20, 3) * 0.5

        return {
            "shap_values": shap_values,
            "feature_names": ["feature1", "feature2", "feature3"],
            "X_test": X_test,
        }

    @pytest.fixture
    def sample_validation_predictions(self):
        """Create sample validation predictions."""
        np.random.seed(42)
        return {
            "y_test": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "y_prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75],
            "test_size": 10,
            "intrusion_rate": 0.5,
        }

    @patch("kedro_project.pipelines.model_interpretability.nodes.shap.TreeExplainer")
    def test_compute_shap_values(self, mock_explainer, sample_model_and_data):
        """Test compute_shap_values function."""
        model, X_test = sample_model_and_data

        # Mock the SHAP explainer
        mock_explainer_instance = Mock()
        mock_explainer.return_value = mock_explainer_instance

        # Mock SHAP values (binary classification returns list of arrays)
        mock_shap_values = [
            np.random.randn(50, 3) * 0.1,  # Negative class
            np.random.randn(50, 3) * 0.1,  # Positive class
        ]
        mock_explainer_instance.shap_values.return_value = mock_shap_values

        result = compute_shap_values(model, X_test)

        # Check that explainer was called
        mock_explainer.assert_called_once_with(model)
        mock_explainer_instance.shap_values.assert_called_once_with(X_test)

        # Check result structure
        assert isinstance(result, dict)
        assert "shap_values" in result
        assert "feature_names" in result
        assert "X_test" in result

        # Check that positive class SHAP values are used
        np.testing.assert_array_equal(result["shap_values"], mock_shap_values[1])
        assert result["feature_names"] == X_test.columns.tolist()
        pd.testing.assert_frame_equal(result["X_test"], X_test)

    @patch("kedro_project.pipelines.model_interpretability.nodes.shap.summary_plot")
    def test_create_shap_summary_plot(self, mock_summary_plot, sample_shap_data):
        """Test create_shap_summary_plot function."""
        fig = create_shap_summary_plot(sample_shap_data)

        # Check that summary plot was called
        mock_summary_plot.assert_called_once()

        # Check that figure is created
        assert isinstance(fig, plt.Figure)

        # Check that title is set
        ax = fig.axes[0]
        assert "SHAP Summary Plot" in ax.get_title()

        # Close figure to prevent memory leaks
        plt.close(fig)

    def test_create_shap_feature_importance(self, sample_shap_data):
        """Test create_shap_feature_importance function."""
        importance_df, fig = create_shap_feature_importance(sample_shap_data)

        # Check importance DataFrame
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 3  # Number of features
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns

        # Check that importance values are sorted
        assert importance_df["importance"].is_monotonic_decreasing

        # Check that all features are present
        assert set(importance_df["feature"].values) == set(
            sample_shap_data["feature_names"]
        )

        # Check figure
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert "SHAP Importance" in ax.get_title()

        # Close figure to prevent memory leaks
        plt.close(fig)

    @patch("kedro_project.pipelines.model_interpretability.nodes.shap.dependence_plot")
    def test_create_shap_dependence_plots(self, mock_dependence_plot, sample_shap_data):
        """Test create_shap_dependence_plots function."""
        key_features = ["feature1", "feature2"]

        plots = create_shap_dependence_plots(sample_shap_data, key_features)

        # Check that plots are created for each feature
        assert isinstance(plots, dict)
        assert "shap_dependence_feature1" in plots
        assert "shap_dependence_feature2" in plots

        # Check that dependence plot was called for each feature
        assert mock_dependence_plot.call_count == 2

        # Check that figures are created
        for plot_name, fig in plots.items():
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_analyze_local_interpretations(
        self, sample_shap_data, sample_validation_predictions
    ):
        """Test analyze_local_interpretations function."""
        results = analyze_local_interpretations(
            sample_shap_data, sample_validation_predictions, n_samples=2
        )

        # Check that results have correct structure
        assert isinstance(results, dict)
        assert "highest_risk" in results
        assert "lowest_risk" in results

        # Check structure of individual interpretations
        for interpretation in results.values():
            assert "index" in interpretation
            assert "predicted_probability" in interpretation
            assert "actual_label" in interpretation
            assert "description" in interpretation
            assert "top_contributing_features" in interpretation

            # Check top contributing features
            top_features = interpretation["top_contributing_features"]
            assert isinstance(top_features, list)
            assert len(top_features) <= 5  # Top 5 features

            for feature_info in top_features:
                assert "feature" in feature_info
                assert "feature_value" in feature_info
                assert "shap_value" in feature_info
                assert "direction" in feature_info
                assert "impact" in feature_info

                # Check that direction is correct
                assert feature_info["direction"] in ["increases", "decreases"]

    def test_create_global_feature_importance_comparison(self):
        """Test create_global_feature_importance_comparison function."""
        # Create sample feature importance data
        model_importance = pd.DataFrame(
            {
                "feature": ["feature1", "feature2", "feature3"],
                "importance": [0.5, 0.3, 0.2],
            }
        )

        shap_importance = pd.DataFrame(
            {
                "feature": ["feature1", "feature2", "feature3"],
                "importance": [0.4, 0.4, 0.2],
            }
        )

        fig = create_global_feature_importance_comparison(
            model_importance, shap_importance
        )

        # Check that figure is created
        assert isinstance(fig, plt.Figure)

        # Check that figure has two subplots
        assert len(fig.axes) == 2

        # Check subplot titles
        ax1, ax2 = fig.axes
        assert "Model Importance" in ax1.get_title()
        assert "SHAP Importance" in ax2.get_title()

        # Close figure to prevent memory leaks
        plt.close(fig)

    def test_shap_dependence_plots_with_missing_features(self, sample_shap_data):
        """Test create_shap_dependence_plots with missing features."""
        # Include a feature that doesn't exist
        key_features = ["feature1", "nonexistent_feature"]

        plots = create_shap_dependence_plots(sample_shap_data, key_features)

        # Should only create plot for existing feature
        assert len(plots) == 1
        assert "shap_dependence_feature1" in plots
        assert "shap_dependence_nonexistent_feature" not in plots

        # Clean up
        for fig in plots.values():
            plt.close(fig)

    def test_local_interpretations_with_edge_cases(self, sample_validation_predictions):
        """Test analyze_local_interpretations with edge cases."""
        # Create minimal SHAP data
        minimal_shap_data = {
            "shap_values": np.array([[0.1, -0.2], [0.3, -0.1]]),
            "feature_names": ["feature1", "feature2"],
            "X_test": pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]}),
        }

        # Create minimal validation predictions
        minimal_predictions = {
            "y_test": [0, 1],
            "y_prob": [0.2, 0.8],
            "test_size": 2,
            "intrusion_rate": 0.5,
        }

        results = analyze_local_interpretations(
            minimal_shap_data, minimal_predictions, n_samples=2
        )

        # Should still return valid results
        assert isinstance(results, dict)
        assert "highest_risk" in results
        assert "lowest_risk" in results

        # Check that indices are different
        assert results["highest_risk"]["index"] != results["lowest_risk"]["index"]

    def test_feature_importance_with_zero_shap_values(self):
        """Test create_shap_feature_importance with zero SHAP values."""
        zero_shap_data = {
            "shap_values": np.zeros((10, 3)),
            "feature_names": ["feature1", "feature2", "feature3"],
            "X_test": pd.DataFrame(
                {
                    "feature1": np.random.randn(10),
                    "feature2": np.random.randn(10),
                    "feature3": np.random.randn(10),
                }
            ),
        }

        importance_df, fig = create_shap_feature_importance(zero_shap_data)

        # Should handle zero values gracefully
        assert len(importance_df) == 3
        assert all(importance_df["importance"] == 0)

        # Close figure
        plt.close(fig)

    def test_json_serialization_of_local_interpretations(
        self, sample_shap_data, sample_validation_predictions
    ):
        """Test that local interpretations are JSON serializable."""
        results = analyze_local_interpretations(
            sample_shap_data, sample_validation_predictions, n_samples=2
        )

        # Check that results can be serialized to JSON
        import json

        try:
            json.dumps(results)
        except TypeError:
            pytest.fail("Local interpretations are not JSON serializable")

    @patch("kedro_project.pipelines.model_interpretability.nodes.shap.TreeExplainer")
    def test_compute_shap_values_with_single_array_output(
        self, mock_explainer, sample_model_and_data
    ):
        """Test compute_shap_values when SHAP returns single array (not list)."""
        model, X_test = sample_model_and_data

        # Mock the SHAP explainer
        mock_explainer_instance = Mock()
        mock_explainer.return_value = mock_explainer_instance

        # Mock SHAP values (single array instead of list)
        mock_shap_values = np.random.randn(50, 3) * 0.1
        mock_explainer_instance.shap_values.return_value = mock_shap_values

        result = compute_shap_values(model, X_test)

        # Check that single array is handled correctly
        assert isinstance(result, dict)
        assert "shap_values" in result
        np.testing.assert_array_equal(result["shap_values"], mock_shap_values)
