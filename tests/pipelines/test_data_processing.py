import pytest
from kedro.pipeline import Pipeline

from kedro_project.pipelines.data_processing import create_pipeline


@pytest.mark.pipeline("data_processing")
def test_data_processing_pipeline():
    """Test that data processing pipeline is created correctly."""
    pipeline = create_pipeline()
    
    # Check that pipeline is created
    assert isinstance(pipeline, Pipeline)
    
    # Check that pipeline has the expected number of nodes
    assert len(pipeline.nodes) == 8
    
    # Check that all expected nodes are present
    node_names = {node.name for node in pipeline.nodes}
    expected_nodes = {
        "load_and_preprocess_data",
        "create_time_features",
        "create_rolling_features",
        "create_lag_features",
        "create_time_since_features",
        "encode_categorical_features",
        "create_feature_matrix",
        "split_time_series_data",
    }
    assert node_names == expected_nodes
    
    # Check pipeline inputs and outputs
    assert "raw_network_logs" in pipeline.inputs()
    assert "X_train" in pipeline.outputs()
    assert "X_test" in pipeline.outputs()
    assert "y_train" in pipeline.outputs()
    assert "y_test" in pipeline.outputs()
    assert "label_encoders" in pipeline.outputs()
    
    # Check that nodes are connected properly
    # First node should take raw_network_logs as input
    first_node = next(n for n in pipeline.nodes if n.name == "load_and_preprocess_data")
    assert "raw_network_logs" in first_node.inputs
    
    # Last node should produce train/test splits
    last_node = next(n for n in pipeline.nodes if n.name == "split_time_series_data")
    assert set(last_node.outputs) == {"X_train", "X_test", "y_train", "y_test"}