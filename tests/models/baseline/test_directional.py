import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


@pytest.mark.model_stage("directional")
def test_directional():
    """Test model responds correctly to directional changes in features."""
    # Create synthetic data with clear directional relationships
    np.random.seed(42)
    n_samples = 200
    
    # Create features where higher values should increase probability of class 1
    X = pd.DataFrame({
        "strong_positive": np.random.randn(n_samples),
        "strong_negative": np.random.randn(n_samples),
        "neutral": np.random.randn(n_samples),
    })
    
    # Create labels with known directional relationships
    # Higher strong_positive -> more likely class 1
    # Higher strong_negative -> more likely class 0
    # neutral has no effect
    y = pd.Series(
        (2 * X["strong_positive"] - 2 * X["strong_negative"] + 0.5 * np.random.randn(n_samples)) > 0
    ).astype(int)
    
    # Train model
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    
    # Test 1: Increasing strong_positive should increase predicted probability
    X_test = pd.DataFrame({
        "strong_positive": [0.0, 1.0, 2.0],
        "strong_negative": [0.0, 0.0, 0.0],
        "neutral": [0.0, 0.0, 0.0],
    })
    probs = model.predict_proba(X_test)[:, 1]
    assert probs[0] < probs[1] < probs[2], "Probability should increase with strong_positive feature"
    
    # Test 2: Increasing strong_negative should decrease predicted probability
    X_test_neg = pd.DataFrame({
        "strong_positive": [0.0, 0.0, 0.0],
        "strong_negative": [0.0, 1.0, 2.0],
        "neutral": [0.0, 0.0, 0.0],
    })
    probs_neg = model.predict_proba(X_test_neg)[:, 1]
    assert probs_neg[0] > probs_neg[1] > probs_neg[2], "Probability should decrease with strong_negative feature"
    
    # Test 3: Feature importance should reflect known relationships
    importance = model.feature_importances_
    feature_names = X.columns.tolist()
    importance_dict = dict(zip(feature_names, importance))
    
    # Strong features should have higher importance than neutral
    assert importance_dict["strong_positive"] > importance_dict["neutral"]
    assert importance_dict["strong_negative"] > importance_dict["neutral"]
