# Network Intrusion Detection System

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-org/network-intrusion-detection)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/downloads/)
[![Kedro](https://img.shields.io/badge/kedro-0.19.14-orange)](https://kedro.org)

A comprehensive machine learning pipeline for detecting network intrusions using time-series network logs. Built with Kedro 0.19.14 and MLflow for experiment tracking.

<!-- Optional: Add project diagram here -->
<!-- ![Network Intrusion Detection Architecture](docs/images/architecture.png) -->

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Setup](#setup)
- [Running Pipelines](#running-pipelines)
- [Experiment Tracking](#experiment-tracking)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project implements a production-ready network intrusion detection system using machine learning techniques. It processes time-series network logs to identify malicious activities and potential security threats. The system leverages Kedro's data pipeline framework for reproducible ML workflows and MLflow for comprehensive experiment tracking and model management.

### Key Technologies

- **Kedro 0.19.14**: Data pipeline framework
- **MLflow**: Experiment tracking and model management
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis

## Features

- **Data Processing Pipeline**: Automated preprocessing of network logs with feature engineering
- **Model Training**: Multiple ML algorithms with hyperparameter optimization
- **Model Validation**: Comprehensive evaluation with cross-validation and metrics
- **Interpretability**: SHAP-based model explanations for security analysts
- **Experiment Tracking**: Full MLflow integration for reproducible experiments
- **Testing Suite**: Comprehensive unit and integration tests

## Project Structure

```
├── conf/                     # Configuration files
│   ├── base/
│   │   ├── catalog.yml      # Data catalog configuration
│   │   ├── parameters.yml   # Pipeline parameters
│   │   └── mlflow.yml       # MLflow configuration
│   └── local/               # Local overrides
├── data/                    # Data storage (Kedro structure)
│   ├── 01_raw/             # Raw network logs
│   ├── 02_intermediate/    # Processed data
│   ├── 03_primary/         # Primary datasets
│   ├── 04_feature/         # Feature engineered data
│   ├── 05_model_input/     # Model training data
│   ├── 06_models/          # Trained models
│   ├── 07_model_output/    # Model predictions
│   └── 08_reporting/       # Reports and visualizations
├── docs/                    # Documentation
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Utility scripts
│   └── start_mlflow_ui.py   # MLflow UI launcher
├── src/kedro_project/       # Source code
│   ├── pipelines/
│   │   ├── data_processing/     # Data preprocessing pipeline
│   │   ├── hyperparameter_optimization/  # Model tuning
│   │   ├── model_validation/    # Model evaluation
│   │   └── model_interpretability/  # Model explanations
│   └── utils/              # Utility functions
└── tests/                  # Test suite
```

## Quick Start

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd network-intrusion-detection
   ```

2. **Set up the environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Start MLflow UI**

   ```bash
   python scripts/start_mlflow_ui.py
   ```

4. **Run the complete pipeline**
   ```bash
   kedro run
   ```

## Setup

### Prerequisites

- Python 3.8 or higher
- Kedro CLI (`pip install kedro`)
- Virtual environment (recommended)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
kedro info
```

### Configuration

The project uses several configuration files:

- **Data Catalog**: [`conf/base/catalog.yml`](conf/base/catalog.yml:1) - Defines data sources and destinations
- **Parameters**: [`conf/base/parameters.yml`](conf/base/parameters.yml:1) - Pipeline parameters and hyperparameters
- **MLflow Config**: [`conf/base/mlflow.yml`](conf/base/mlflow.yml:1) - Experiment tracking settings

### Environment Variables

Set the following environment variables:

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
```

## Running Pipelines

### Complete Pipeline

Run the entire pipeline from data processing to model validation:

```bash
kedro run
```

### Individual Pipelines

#### Data Processing

Preprocess raw network logs and engineer features:

```bash
kedro run --pipeline data_processing
```

#### Hyperparameter Optimization

Optimize model hyperparameters:

```bash
kedro run --pipeline hyperparameter_optimization
```

#### Model Validation

Validate trained models:

```bash
kedro run --pipeline model_validation
```

#### Model Interpretability

Generate model explanations:

```bash
kedro run --pipeline model_interpretability
```

### Pipeline with Parameters

Override default parameters:

```bash
kedro run --params model_type:random_forest,max_depth:10
```

### Pipeline Visualization

Generate pipeline visualization:

```bash
kedro viz
```

## Experiment Tracking

### MLflow Setup

The project uses MLflow for experiment tracking. The MLflow backend is configured to use SQLite (`mlflow.db`).

### Starting MLflow UI

```bash
# Start MLflow UI on port 5000
python scripts/start_mlflow_ui.py

# Or manually
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

### Accessing Experiments

1. Navigate to http://localhost:5000
2. View experiments, runs, metrics, and artifacts
3. Compare model performance across different runs

### Logging Custom Metrics

Metrics are automatically logged during pipeline execution. Key metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Feature importance scores

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Suites

```bash
# Data processing tests
pytest tests/pipelines/data_processing/

# Model validation tests
pytest tests/pipelines/model_validation/

# Hyperparameter optimization tests
pytest tests/pipelines/hyperparameter_optimization/
```

### Run Tests with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Tests in Parallel

```bash
pytest -n auto
```

## Documentation

### Generate Documentation

```bash
kedro build-docs
```

### View Documentation

```bash
# After building docs
open docs/build/html/index.html
```

### API Documentation

API documentation is available in the `docs/` directory and includes:

- Pipeline documentation
- Node-level documentation
- Configuration reference

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. **Run pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Write comprehensive tests for new features

### Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add tests for new functionality
4. Update CHANGELOG.md
5. Submit pull request with clear description

### Reporting Issues

Use GitHub issues to report bugs or request features. Include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

## License

This project follows standard Kedro project licensing. Refer to [LICENSE](LICENSE) file for specific license details.

---

## Additional Resources

- [Kedro Documentation](https://docs.kedro.org)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Project Wiki](https://github.com/your-org/network-intrusion-detection/wiki)

## Support

For questions or support, please:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/your-org/network-intrusion-detection/issues)
3. Create a new issue with the "question" label
