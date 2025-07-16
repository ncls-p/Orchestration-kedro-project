"""MLflow utilities for model registry and tracking."""
import logging
from typing import Any, Dict, Optional

import mlflow
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def register_model(
    model_name: str,
    model_uri: str,
    stage: str = "Staging",
    description: Optional[str] = None,
) -> None:
    """Register a model in MLflow model registry.
    
    Args:
        model_name: Name of the model to register
        model_uri: URI of the model to register
        stage: Stage to transition the model to
        description: Description of the model version
    """
    try:
        client = MlflowClient()
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={"project": "network-intrusion-detection"}
        )
        
        # Add description if provided
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        # Transition to stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )
        
        logger.info(f"Model {model_name} v{model_version.version} registered and transitioned to {stage}")
        
    except Exception as e:
        logger.error(f"Failed to register model {model_name}: {e}")


def get_model_from_registry(
    model_name: str,
    stage: str = "Production"
) -> Any:
    """Load a model from MLflow model registry.
    
    Args:
        model_name: Name of the registered model
        stage: Stage of the model to load
        
    Returns:
        Loaded model
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.lightgbm.load_model(model_uri)
        logger.info(f"Loaded model {model_name} from {stage} stage")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name} from {stage}: {e}")
        raise


def log_model_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    """Log model metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        prefix: Prefix to add to metric names
    """
    try:
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"{prefix}{metric_name}", metric_value)
                
        logger.info(f"Logged {len(metrics)} metrics to MLflow")
        
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {e}")


def start_mlflow_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> mlflow.ActiveRun:
    """Start an MLflow run with experiment and tags.
    
    Args:
        experiment_name: Name of the experiment
        run_name: Name of the run
        tags: Tags to add to the run
        
    Returns:
        Active MLflow run
    """
    try:
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        
        return run
        
    except Exception as e:
        logger.error(f"Failed to start MLflow run: {e}")
        raise