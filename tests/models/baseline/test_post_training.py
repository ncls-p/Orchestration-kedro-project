import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


@pytest.mark.model_stage("post_training")
def test_post_training():
    """Test model evaluation metrics after training."""
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(200),
        "feature2": np.random.randn(200),
        "feature3": np.random.randn(200),
    })
    # Create labels with some correlation to features
    y = pd.Series((X["feature1"] + X["feature2"] + np.random.randn(200) * 0.5) > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    model = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Test that metrics are within reasonable ranges
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0
    
    # Test that model performs better than random
    assert accuracy > 0.5
    
    # Test probability calibration
    # Average predicted probability should be close to actual positive rate
    predicted_positive_rate = y_proba.mean()
    actual_positive_rate = y_test.mean()
    assert abs(predicted_positive_rate - actual_positive_rate) < 0.3
    
    # Test that probabilities are in valid range
    assert all(0.0 <= p <= 1.0 for p in y_proba)
