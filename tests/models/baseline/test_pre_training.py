import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@pytest.mark.model_stage("pre_training")
def test_pre_training():
    """Test data preprocessing steps before model training."""
    # Create sample data with various edge cases
    np.random.seed(42)
    
    # Test data with missing values
    X = pd.DataFrame({
        "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feature2": [10, 20, 30, 40, 50],
        "feature3": ["A", "B", "A", "C", "B"],
    })
    y = pd.Series([0, 1, 0, 1, 1])
    
    # Test handling of missing values
    assert X.isnull().any().any()  # Confirm we have missing values
    X_filled = X.copy()
    X_filled["feature1"] = X_filled["feature1"].fillna(X_filled["feature1"].mean())
    assert not X_filled.isnull().any().any()  # No more missing values
    
    # Test data scaling
    scaler = StandardScaler()
    numeric_features = ["feature1", "feature2"]
    X_numeric = X_filled[numeric_features]
    X_scaled = scaler.fit_transform(X_numeric)
    
    # Check that scaling produces mean ~0 and std ~1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)
    
    # Test class balance check
    class_counts = y.value_counts()
    assert len(class_counts) == 2  # Binary classification
    imbalance_ratio = class_counts.max() / class_counts.min()
    assert imbalance_ratio < 3  # Not severely imbalanced
    
    # Test data shape consistency
    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert X.shape[1] > 0
