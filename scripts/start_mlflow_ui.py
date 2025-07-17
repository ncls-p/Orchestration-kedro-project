#!/usr/bin/env python3
"""Start MLflow UI with correct configuration."""

import logging
import os
import subprocess
import sys

LOGGER = logging.getLogger(__name__)


def start_mlflow_ui():
    """Start MLflow UI with SQLite backend."""
    # Check if we're in the right directory
    if not os.path.exists("mlflow.db"):
        LOGGER.info(
            "Error: mlflow.db not found. Please run this script from the project root."
        )
        sys.exit(1)

    # MLflow UI command
    cmd = [
        "mlflow",
        "ui",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--port",
        "5000",
        "--host",
        "127.0.0.1",
    ]

    LOGGER.info("Starting MLflow UI...")
    LOGGER.info("Command: %s", " ".join(cmd))
    LOGGER.info("UI will be available at: http://127.0.0.1:5000")
    LOGGER.info("Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        LOGGER.info("MLflow UI stopped.")
    except subprocess.CalledProcessError as e:
        LOGGER.info("Error starting MLflow UI: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    start_mlflow_ui()
