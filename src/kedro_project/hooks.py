"""Project hooks for the kedro-project.

This module contains Kedro hooks that provide project-wide functionality including
MLflow experiment setup and pipeline lifecycle management. Hooks are automatically
discovered and executed by Kedro at appropriate points in the pipeline lifecycle.

The MLflow integration logs:
- Hyperparameters from optimization
- Model artifacts and metadata
- Performance metrics (F1, precision, recall, accuracy, AUC)
- Training data statistics
- Visualizations and plots
- Model feature importance
- SHAP interpretability results
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.lightgbm as mlflow_lightgbm
import numpy as np
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)


class _LogMessage:
    def __init__(self, msg: str):
        self.message = msg

    def __str__(self):
        return self.message


class MLflowIntegrationHook:
    """Custom MLflow hook for network intrusion detection project."""

    def __init__(self):
        self.experiment_name = "network-intrusion-detection"
        self.setup_mlflow()

    def setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Set tracking URI to SQLite database
            tracking_uri = "sqlite:///mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")

            # Enable automatic logging for system metrics
            mlflow.autolog(
                log_input_examples=False,  # Disable to avoid large artifacts
                log_model_signatures=False,  # Disable to avoid conflicts
                log_models=False,  # We handle model logging manually
                silent=True  # Reduce verbose output
            )

            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                    logger.info(f"Created MLflow experiment: {self.experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    logger.info(
                        f"Using existing MLflow experiment: {self.experiment_name}"
                    )
            except Exception as e:
                logger.warning(f"Error setting up MLflow experiment: {e}")
                experiment_id = None

            if experiment_id:
                mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")

    @hook_impl
    def before_node_run(
        self, node: Node, catalog: DataCatalog, inputs: dict[str, Any], is_async: bool
    ) -> None:
        """Hook to be called before a node runs."""
        # Log start of critical ML nodes
        if node.name in [
            "optimize_hyperparameters",
            "train_optimized_model",
            "evaluate_model",
            "compute_shap_values",
            "create_roc_curve",
            "create_pr_curve"
        ]:
            logger.info(f"Starting MLflow tracking for node: {node.name}")

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Hook to be called after a node runs."""
        try:
            # Core ML workflow logging
            if node.name == "optimize_hyperparameters":
                self._log_hyperparameter_optimization(outputs)

            elif node.name == "train_optimized_model":
                self._log_model_training(inputs, outputs, catalog)

            elif node.name == "evaluate_model":
                self._log_model_evaluation(outputs)

            # Visualization and interpretability logging
            elif node.name == "create_roc_curve":
                self._log_roc_curve(outputs)

            elif node.name == "create_pr_curve":
                self._log_pr_curve(outputs)

            elif node.name == "compute_shap_values":
                self._log_shap_artifacts(outputs)

            elif node.name in [
                "create_shap_summary_plot",
                "create_shap_feature_importance",
                "create_shap_dependence_plots",
                "create_global_feature_importance_comparison",
                "create_density_chart",
                "create_calibration_curve"
            ]:
                self._log_visualization_artifacts(node.name, outputs)

        except Exception as e:
            logger.error(f"Error in MLflow logging for node {node.name}: {e}")

    def _log_hyperparameter_optimization(self, outputs: dict[str, Any]) -> None:
        """Log hyperparameter optimization results."""
        if "optimized_hyperparameters" in outputs:
            hyperparams = outputs["optimized_hyperparameters"]
            
            try:
                # Get the parent run to log parameters there
                active_run = mlflow.active_run()
                if active_run:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Check if this is a nested run by looking for parent run ID in tags
                    parent_run_id = active_run.data.tags.get('mlflow.parentRunId', None)
                    
                    if parent_run_id:
                        # Log best hyperparameters to parent run
                        for param_name, param_value in hyperparams.items():
                            client.log_param(parent_run_id, f"best_{param_name}", param_value)
                            
                        # Add tags to parent run
                        client.set_tag(parent_run_id, "stage", "hyperparameter_optimization")
                        client.set_tag(parent_run_id, "model_type", "lightgbm")
                        client.set_tag(parent_run_id, "optimization_method", "hyperopt_tpe")
                        
                        logger.info(f"Logged hyperparameter optimization results to parent MLflow run {parent_run_id}")
                    else:
                        # If no parent run, log to current run
                        for param_name, param_value in hyperparams.items():
                            mlflow.log_param(f"best_{param_name}", param_value)
                            
                        # Add tags to current run
                        mlflow.set_tag("stage", "hyperparameter_optimization")
                        mlflow.set_tag("model_type", "lightgbm")
                        mlflow.set_tag("optimization_method", "hyperopt_tpe")
                        
                        logger.info("Logged hyperparameter optimization results to current MLflow run")
                else:
                    logger.warning("No active MLflow run found")
            except Exception as e:
                logger.error(f"Failed to log hyperparameter optimization: {e}")

    def _log_model_training(
        self, inputs: dict[str, Any], outputs: dict[str, Any], catalog: DataCatalog
    ) -> None:
        """Log model training results."""
        if "trained_model" in outputs:
            try:
                model = outputs["trained_model"]
                
                # Get the parent run to log parameters there
                active_run = mlflow.active_run()
                if active_run:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Check if this is a nested run by looking for parent run ID in tags
                    parent_run_id = active_run.data.tags.get('mlflow.parentRunId', None)
                    
                    if parent_run_id:
                        # Add run tags to parent run
                        client.set_tag(parent_run_id, "stage", "model_training")
                        client.set_tag(parent_run_id, "model_type", "lightgbm")
                        client.set_tag(parent_run_id, "algorithm", "gradient_boosting")

                        # Log model parameters to parent run
                        if "optimized_hyperparameters" in inputs:
                            hyperparams = inputs["optimized_hyperparameters"]
                            for param_name, param_value in hyperparams.items():
                                client.log_param(parent_run_id, param_name, param_value)

                        # Log feature importance if available
                        if hasattr(model, "feature_importances_"):
                            feature_importance = model.feature_importances_
                            client.log_metric(parent_run_id, "n_features", len(feature_importance))
                            client.log_metric(parent_run_id, "max_feature_importance", float(np.max(feature_importance)))
                            client.log_metric(parent_run_id, "mean_feature_importance", float(np.mean(feature_importance)))

                        logger.info(f"Logged model training results to parent MLflow run {parent_run_id}")
                    else:
                        # If no parent run, log to current run
                        mlflow.set_tag("stage", "model_training")
                        mlflow.set_tag("model_type", "lightgbm")
                        mlflow.set_tag("algorithm", "gradient_boosting")

                        # Log model parameters
                        if "optimized_hyperparameters" in inputs:
                            hyperparams = inputs["optimized_hyperparameters"]
                            for param_name, param_value in hyperparams.items():
                                mlflow.log_param(param_name, param_value)

                        # Log feature importance if available
                        if hasattr(model, "feature_importances_"):
                            feature_importance = model.feature_importances_
                            mlflow.log_metric("n_features", len(feature_importance))
                            mlflow.log_metric("max_feature_importance", float(np.max(feature_importance)))
                            mlflow.log_metric("mean_feature_importance", float(np.mean(feature_importance)))

                        logger.info("Logged model training results to current MLflow run")

                # Log training data statistics (always to current run)
                self._log_training_data_stats(inputs, catalog)

                # Log model using LightGBM flavor (always to current run)
                try:
                    mlflow_lightgbm.log_model(
                        model,
                        "model",
                        registered_model_name="network-intrusion-detection-model",
                    )
                    logger.info("Logged trained model to MLflow")
                except Exception as exc:
                    logger.error(f"Failed to log model: {exc}")
                    
            except Exception as e:
                logger.error(f"Failed to log model training: {e}")

    def _log_model_evaluation(self, outputs: dict[str, Any]) -> None:
        """Log model evaluation metrics."""
        if "model_performance_metrics" in outputs:
            metrics = outputs["model_performance_metrics"]
            
            try:
                # Get the parent run to log metrics there
                active_run = mlflow.active_run()
                if active_run:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Check if this is a nested run by looking for parent run ID in tags
                    parent_run_id = active_run.data.tags.get('mlflow.parentRunId', None)
                    
                    if parent_run_id:
                        # Log metrics to parent run
                        metric_names = ["f1_score", "precision", "recall", "accuracy", "roc_auc", "pr_auc", "cv_mean_accuracy", "cv_std_accuracy"]
                        for metric_name in metric_names:
                            if metric_name in metrics:
                                client.log_metric(parent_run_id, metric_name, metrics[metric_name])

                        # Log confusion matrix if available
                        if "confusion_matrix" in metrics:
                            cm = metrics["confusion_matrix"]
                            client.log_metric(parent_run_id, "true_negatives", int(cm[0][0]))
                            client.log_metric(parent_run_id, "false_positives", int(cm[0][1]))
                            client.log_metric(parent_run_id, "false_negatives", int(cm[1][0]))
                            client.log_metric(parent_run_id, "true_positives", int(cm[1][1]))

                        # Add evaluation tags to parent run
                        client.set_tag(parent_run_id, "stage", "model_evaluation")
                        client.set_tag(parent_run_id, "evaluation_type", "test_set")
                        
                        logger.info(f"Logged model evaluation metrics to parent MLflow run {parent_run_id}")
                    else:
                        # If no parent run, log to current run
                        metric_names = ["f1_score", "precision", "recall", "accuracy", "roc_auc", "pr_auc", "cv_mean_accuracy", "cv_std_accuracy"]
                        for metric_name in metric_names:
                            if metric_name in metrics:
                                mlflow.log_metric(metric_name, metrics[metric_name])

                        # Log confusion matrix if available
                        if "confusion_matrix" in metrics:
                            cm = metrics["confusion_matrix"]
                            mlflow.log_metric("true_negatives", int(cm[0][0]))
                            mlflow.log_metric("false_positives", int(cm[0][1]))
                            mlflow.log_metric("false_negatives", int(cm[1][0]))
                            mlflow.log_metric("true_positives", int(cm[1][1]))

                        # Add evaluation tags
                        mlflow.set_tag("stage", "model_evaluation")
                        mlflow.set_tag("evaluation_type", "test_set")
                        
                        logger.info("Logged model evaluation metrics to current MLflow run")
                else:
                    logger.warning("No active MLflow run found")
            except Exception as e:
                logger.error(f"Failed to log model evaluation metrics: {e}")

    def _log_training_data_stats(
        self, inputs: dict[str, Any], catalog: DataCatalog
    ) -> None:
        """Log training data statistics."""
        try:
            # Get training data info
            if "X_train" in inputs and "y_train" in inputs:
                X_train = inputs["X_train"]
                y_train = inputs["y_train"]

                # Get the parent run to log parameters there
                active_run = mlflow.active_run()
                if active_run:
                    client = mlflow.tracking.MlflowClient()
                    
                    # Check if this is a nested run by looking for parent run ID in tags
                    parent_run_id = active_run.data.tags.get('mlflow.parentRunId', None)
                    
                    if parent_run_id:
                        # Log data shapes to parent run
                        client.log_param(parent_run_id, "n_train_samples", len(X_train))
                        client.log_param(parent_run_id, "n_features", X_train.shape[1] if hasattr(X_train, 'shape') else 0)

                        # Log target distribution to parent run
                        if hasattr(y_train, 'value_counts'):
                            class_counts = y_train.value_counts()
                            for class_label, count in class_counts.items():
                                client.log_metric(parent_run_id, f"class_{class_label}_count", count)
                            client.log_metric(parent_run_id, "class_balance_ratio",
                                            float(class_counts.min() / class_counts.max()))
                        
                        logger.info(f"Logged training data stats to parent MLflow run {parent_run_id}")
                    else:
                        # If no parent run, log to current run
                        mlflow.log_param("n_train_samples", len(X_train))
                        mlflow.log_param("n_features", X_train.shape[1] if hasattr(X_train, 'shape') else 0)

                        # Log target distribution
                        if hasattr(y_train, 'value_counts'):
                            class_counts = y_train.value_counts()
                            for class_label, count in class_counts.items():
                                mlflow.log_metric(f"class_{class_label}_count", count)
                            mlflow.log_metric("class_balance_ratio",
                                            float(class_counts.min() / class_counts.max()))
                        
                        logger.info("Logged training data stats to current MLflow run")

        except Exception as e:
            logger.warning(f"Could not log training data stats: {e}")

    def _log_roc_curve(self, outputs: dict[str, Any]) -> None:
        """Log ROC curve artifact."""
        if "roc_curve" in outputs:
            try:
                mlflow.set_tag("stage", "validation")
                mlflow.set_tag("visualization_type", "roc_curve")

                # Log ROC curve plot as artifact
                plot_path = "roc_curve.png"
                outputs["roc_curve"].write_image(plot_path)
                mlflow.log_artifact(plot_path)
                # Clean up temp file
                if os.path.exists(plot_path):
                    os.remove(plot_path)
                logger.info("Logged ROC curve to MLflow")
            except Exception as e:
                logger.warning(f"Could not log ROC curve: {e}")

    def _log_pr_curve(self, outputs: dict[str, Any]) -> None:
        """Log Precision-Recall curve artifact."""
        if "pr_curve" in outputs:
            try:
                mlflow.set_tag("stage", "validation")
                mlflow.set_tag("visualization_type", "precision_recall_curve")

                # Log PR curve plot as artifact
                plot_path = "pr_curve.png"
                outputs["pr_curve"].write_image(plot_path)
                mlflow.log_artifact(plot_path)
                # Clean up temp file
                if os.path.exists(plot_path):
                    os.remove(plot_path)
                logger.info("Logged PR curve to MLflow")
            except Exception as e:
                logger.warning(f"Could not log PR curve: {e}")

    def _log_shap_artifacts(self, outputs: dict[str, Any]) -> None:
        """Log SHAP values as artifacts."""
        if "shap_data" in outputs:
            try:
                mlflow.set_tag("stage", "interpretability")
                mlflow.set_tag("interpretation_method", "shap")

                shap_data = outputs["shap_data"]

                # Log SHAP statistics
                if "shap_values" in shap_data:
                    shap_values = shap_data["shap_values"]
                    mlflow.log_metric("shap_values_mean_abs", float(np.mean(np.abs(shap_values))))
                    mlflow.log_metric("shap_values_max_abs", float(np.max(np.abs(shap_values))))

                logger.info("Logged SHAP analysis to MLflow")
            except Exception as e:
                logger.warning(f"Could not log SHAP analysis: {e}")

    def _log_visualization_artifacts(self, node_name: str, outputs: dict[str, Any]) -> None:
        """Log visualization artifacts for various plot types."""
        # Map node names to output keys and plot types
        plot_mappings = {
            "create_shap_summary_plot": ("shap_summary_plot", "shap_summary"),
            "create_shap_feature_importance": ("shap_feature_importance_plot", "shap_importance"),
            "create_shap_dependence_plots": ("shap_dependence_plots", "shap_dependence"),
            "create_global_feature_importance_comparison": ("feature_importance_comparison_plot", "feature_comparison"),
            "create_density_chart": ("density_chart", "density_plot"),
            "create_calibration_curve": ("calibration_curve", "calibration")
        }

        if node_name in plot_mappings:
            output_key, plot_type = plot_mappings[node_name]

            if output_key in outputs:
                try:
                    mlflow.set_tag("stage", "visualization")
                    mlflow.set_tag("visualization_type", plot_type)

                    # Log plot as artifact
                    plot_path = f"{plot_type}.png"
                    if hasattr(outputs[output_key], 'write_image'):
                        outputs[output_key].write_image(plot_path)
                    elif hasattr(outputs[output_key], 'savefig'):
                        outputs[output_key].savefig(plot_path)
                    mlflow.log_artifact(plot_path)
                    # Clean up temp file
                    if os.path.exists(plot_path):
                        os.remove(plot_path)
                    logger.info(f"Logged {plot_type} visualization to MLflow")
                except Exception as e:
                    logger.warning(f"Could not log {plot_type} visualization: {e}")
