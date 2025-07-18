import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


@pytest.mark.model_stage("invariance")
def test_invariance():
    """Test model invariance to irrelevant transformations."""
    # Create synthetic data
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
    })
    y = pd.Series(np.random.choice([0, 1], size=100))
    
    # Train model
    model = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    
    # Test 1: Model should be invariant to feature order
    X_reordered = X[["feature3", "feature1", "feature2"]]
    pred_original_all = model.predict_proba(X)
    pred_original = np.asarray(pred_original_all)[:, 1]
    
    # Retrain model with reordered features
    model_reordered = LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model_reordered.fit(X_reordered, y)
    pred_reordered_all = model_reordered.predict_proba(X_reordered)
    pred_reordered = np.asarray(pred_reordered_all)[:, 1]
    
    # Predictions should be very similar
    assert np.allclose(pred_original, pred_reordered, atol=0.1)
    
    # Test 2: Model should be robust to small perturbations
    X_perturbed = X + np.random.normal(0, 0.01, X.shape)
    pred_perturbed_all = model.predict_proba(X_perturbed)
    pred_perturbed = np.asarray(pred_perturbed_all)[:, 1]
    
    # Most predictions should remain similar
    prediction_changes = np.abs(pred_original - pred_perturbed)
    assert np.mean(prediction_changes < 0.1) > 0.9  # 90% should change by less than 0.1
    
    # Test 3: Model should handle duplicate samples consistently
    X_duplicated = pd.concat([X, X.iloc[[0]]], ignore_index=True)
    pred_duplicated_all = model.predict_proba(X_duplicated)
    pred_duplicated = np.asarray(pred_duplicated_all)[:, 1]
    
    # First and last predictions should be identical (same sample)
    assert np.abs(pred_duplicated[0] - pred_duplicated[-1]) < 1e-6
