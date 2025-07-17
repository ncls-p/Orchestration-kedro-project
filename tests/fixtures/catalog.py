import pytest


@pytest.fixture
def dummy_memory_data_set():
    """Provide a dummy in-memory dataset."""
    return {"data": "dummy_data"}


@pytest.fixture
def catalog_dict(dummy_memory_data_set):
    """Return a minimal catalog dictionary."""
    return {"dummy": dummy_memory_data_set}
