import pytest
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score


@pytest.mark.model_stage("unit")
def test_unit():
    """Unit test for basic model functionality."""
    # Create simple synthetic data
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
    })
    y = pd.Series(np.random.choice([0, 1], size=100))
    
    # Test model initialization
    model = LGBMClassifier(
        n_estimators=10,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    
    # Test that model can be trained
    model.fit(X, y)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    
    # Test predictions
    predictions = model.predict(X)
    assert predictions is not None
    # Convert to numpy array to ensure consistent behavior
    pred_array = np.asarray(predictions)
    assert pred_array.shape[0] == len(y)
    assert np.all(np.isin(pred_array, [0, 1]))
    
    # Test probability predictions
    probabilities = model.predict_proba(X)
    assert probabilities is not None
    # Convert to numpy array to ensure consistent behavior
    prob_array = np.asarray(probabilities)
    assert prob_array.shape == (len(X), 2)
    assert np.allclose(prob_array.sum(axis=1), 1.0)
    
    # Test that model achieves better than random performance on training data
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.5  # Better than random guessing
