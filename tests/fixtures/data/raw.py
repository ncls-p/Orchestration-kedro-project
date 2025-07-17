import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    """Return a minimal raw dataframe."""
    return pd.DataFrame({"x": [1, 2], "y": [3, 4]})
