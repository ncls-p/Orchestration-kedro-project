import logging
from typing import Any, Dict

import mlflow
import mlflow.lightgbm
from kedro.framework.context import KedroContext
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)


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
        self, node: Node, catalog: DataCatalog, inputs: Dict[str, Any], is_async: bool
    ) -> None:
        """Hook to be called before a node runs."""
        if node.name == "optimize_hyperparameters":
            logger.info("Starting hyperparameter optimization MLflow run")

        elif node.name == "train_optimized_model":
            logger.info("Starting model training MLflow run")

        elif node.name == "evaluate_model":
            logger.info("Starting model evaluation MLflow run")

    @hook_impl
    def after_node_run(
        self,
        node: Node,
        catalog: DataCatalog,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        """Hook to be called after a node runs."""
        try:
            if node.name == "optimize_hyperparameters":
                self._log_hyperparameter_optimization(outputs)

            elif node.name == "train_optimized_model":
                self._log_model_training(inputs, outputs)

            elif node.name == "evaluate_model":
                self._log_model_evaluation(outputs)

        except Exception as e:
            logger.error(f"Error in MLflow logging for node {node.name}: {e}")

    def _log_hyperparameter_optimization(self, outputs: Dict[str, Any]) -> None:
        """Log hyperparameter optimization results."""
        if "optimized_hyperparameters" in outputs:
            with mlflow.start_run(nested=True, run_name="hyperparameter_optimization"):
                hyperparams = outputs["optimized_hyperparameters"]

                # Log hyperparameters
                for param_name, param_value in hyperparams.items():
                    mlflow.log_param(f"best_{param_name}", param_value)

                logger.info("Logged hyperparameter optimization results to MLflow")

    def _log_model_training(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> None:
        """Log model training results."""
        if "trained_model" in outputs:
            with mlflow.start_run(nested=True, run_name="model_training"):
                model = outputs["trained_model"]

                # Log model parameters
                if "optimized_hyperparameters" in inputs:
                    hyperparams = inputs["optimized_hyperparameters"]
                    for param_name, param_value in hyperparams.items():
                        mlflow.log_param(param_name, param_value)

                # Log model using LightGBM flavor
                try:
                    mlflow.lightgbm.log_model(
                        model,
                        "model",
                        registered_model_name="network-intrusion-detection-model",
                    )
                    logger.info("Logged trained model to MLflow")
                except Exception as e:
                    logger.error(f"Failed to log model to MLflow: {e}")

    def _log_model_evaluation(self, outputs: Dict[str, Any]) -> None:
        """Log model evaluation metrics."""
        if "model_performance_metrics" in outputs:
            with mlflow.start_run(nested=True, run_name="model_evaluation"):
                metrics = outputs["model_performance_metrics"]

                # Log performance metrics
                if "f1_score" in metrics:
                    mlflow.log_metric("f1_score", metrics["f1_score"])
                if "precision" in metrics:
                    mlflow.log_metric("precision", metrics["precision"])
                if "recall" in metrics:
                    mlflow.log_metric("recall", metrics["recall"])

                logger.info("Logged model evaluation metrics to MLflow")
