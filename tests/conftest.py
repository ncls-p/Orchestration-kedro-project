"""
Shared fixtures and configuration for tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


@pytest.fixture(scope="session")
def project_path():
    """Return the path to the Kedro project."""
    return Path.cwd()


@pytest.fixture(scope="session")
def kedro_session(project_path):
    """Create a Kedro session for testing."""
    bootstrap_project(project_path)
    with KedroSession.create(project_path=project_path) as session:
        yield session


@pytest.fixture
def sample_network_data():
    """Create sample network data for testing."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Timestamp": pd.date_range(start="2025-01-01", periods=100, freq="H"),
            "Source_IP": [f"192.168.1.{i % 10}" for i in range(100)],
            "Destination_IP": [f"10.0.0.{i % 5}" for i in range(100)],
            "Port": np.random.choice([80, 443, 8080, 3306], size=100),
            "Request_Type": np.random.choice(
                ["GET", "POST", "PUT", "DELETE"], size=100
            ),
            "Protocol": np.random.choice(["HTTP", "HTTPS", "TCP", "UDP"], size=100),
            "Payload_Size": np.random.randint(100, 3000, size=100),
            "User_Agent": np.random.choice(
                ["Mozilla/5.0", "Chrome/91.0", "Safari/14.0"], size=100
            ),
            "Status": np.random.choice(["Success", "Error", "Timeout"], size=100),
            "Intrusion": np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
            "Scan_Type": np.random.choice(
                ["Normal", "Port_Scan", "Brute_Force"], size=100
            ),
        }
    )

    return data


@pytest.fixture
def sample_feature_matrix():
    """Create sample feature matrix for testing."""
    np.random.seed(42)

    return pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "feature4": np.random.randint(0, 5, size=100),
            "feature5": np.random.uniform(0, 1, size=100),
        }
    )


@pytest.fixture
def sample_target_vector():
    """Create sample target vector for testing."""
    np.random.seed(42)
    return pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))


@pytest.fixture
def sample_model_specs():
    """Create sample model specifications for testing."""
    return {
        "name": "LightGBM",
        "max_evals": 5,  # Small number for testing
        "params": {
            "learning_rate": {"type": "uniform", "low": 0.01, "high": 0.3},
            "num_iterations": {"type": "quniform", "low": 50, "high": 200, "q": 50},
            "max_depth": {"type": "quniform", "low": 3, "high": 8, "q": 1},
        },
        "override_schemas": {"num_iterations": "int", "max_depth": "int"},
    }


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress warnings during testing."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def matplotlib_cleanup():
    """Clean up matplotlib figures after tests."""
    import matplotlib.pyplot as plt

    yield

    # Close all figures to prevent memory leaks
    plt.close("all")


# ---------------------------------------------------------------------------
# Auto-discovered plugin modules and shared marker constants for new test-suite
# ---------------------------------------------------------------------------
pytest_plugins = [
    "tests.fixtures.data.raw",
    "tests.fixtures.data.processed",
    "tests.fixtures.models.base",
    "tests.fixtures.models.trained",
    "tests.fixtures.catalog",
]

PIPELINE_MARKER = "pipeline"
MODEL_STAGE_MARKER = "model_stage"
