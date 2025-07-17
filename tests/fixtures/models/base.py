import pytest
from sklearn.dummy import DummyRegressor


@pytest.fixture
def untrained_model():
    """Return an untrained dummy regressor."""
    return DummyRegressor(strategy="mean")
