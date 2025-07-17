import pandas as pd
import pytest


@pytest.fixture
def processed_df():
    """Return a minimal processed dataframe."""
    return pd.DataFrame({"x": [10, 20], "y": [30, 40]})
