"""Project settings for the kedro-project.

This module contains the main configuration settings for the Kedro project,
including hook registration, configuration loader setup, and project-wide
settings that control pipeline behavior and plugin integration.

For further information about Kedro settings, see:
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html
"""

# Instantiated project hooks.
# from kedro_project.hooks import SparkHooks  # noqa: E402
from kedro_mlflow.framework.hooks import MlflowHook

from kedro_project.hooks import MLflowIntegrationHook

# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (MlflowHook(), MLflowIntegrationHook())
"""Tuple of instantiated project hooks.

This tuple contains instances of hook classes that will be automatically
executed by Kedro at appropriate points in the pipeline lifecycle. The hooks
provide cross-cutting functionality like MLflow experiment setup and logging.

The hooks are executed in Last-In-First-Out (LIFO) order, meaning the last
hook in the tuple is executed first. This allows for proper initialization
and cleanup sequencing.

Current hooks:
- MlflowHook: From kedro-mlflow plugin for MLflow integration
- MLflowIntegrationHook: Custom hook for project-specific MLflow tracking
"""

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader  # noqa: E402

CONFIG_LOADER_CLASS = OmegaConfigLoader
"""Class that manages how configuration is loaded.

Uses OmegaConfigLoader for advanced configuration management with support
for environment-specific configurations and pattern-based config loading.
"""

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
        "mlflow": ["mlflow*", "mlflow*/**"],
    },
}
"""Configuration loader arguments.

Dictionary containing configuration for the OmegaConfigLoader including:
- base_env: The base environment for configuration loading
- default_run_env: The default environment to use when running pipelines
- config_patterns: Pattern matching for different configuration types
"""

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
