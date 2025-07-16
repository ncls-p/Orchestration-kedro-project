#!/usr/bin/env python3
"""Start MLflow UI with correct configuration."""

import os
import subprocess
import sys


def start_mlflow_ui():
    """Start MLflow UI with SQLite backend."""
    # Check if we're in the right directory
    if not os.path.exists("mlflow.db"):
        print(
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

    print("Starting MLflow UI...")
    print(f"Command: {' '.join(cmd)}")
    print("UI will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow UI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_mlflow_ui()
