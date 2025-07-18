import pytest
from kedro.pipeline import Pipeline

from kedro_project.pipelines.hyperparameter_optimization import create_pipeline


@pytest.mark.pipeline("hyperparameter_optimization")
def test_hyperparameter_optimization_pipeline():
    """Test that hyperparameter optimization pipeline is created correctly."""
    pipeline = create_pipeline()
    
    # Check that pipeline is created
    assert isinstance(pipeline, Pipeline)
    
    # Check that pipeline has the expected number of nodes
    assert len(pipeline.nodes) == 2
    
    # Check that all expected nodes are present
    node_names = {node.name for node in pipeline.nodes}
    expected_nodes = {
        "optimize_hyperparameters",
        "train_optimized_model",
    }
    assert node_names == expected_nodes
    
    # Check pipeline inputs and outputs
    assert "X_train" in pipeline.inputs()
    assert "y_train" in pipeline.inputs()
    assert "trained_model" in pipeline.outputs()
    # Note: optimized_hyperparameters is an intermediate output, not exposed
    
    # Check that nodes are connected properly
    # First node should optimize hyperparameters
    first_node = next(n for n in pipeline.nodes if n.name == "optimize_hyperparameters")
    assert "X_train" in first_node.inputs
    assert "y_train" in first_node.inputs
    assert "params:model_specs" in first_node.inputs
    assert "params:cv_folds" in first_node.inputs
    assert "optimized_hyperparameters" in first_node.outputs
    
    # Second node should train the model with optimized parameters
    second_node = next(n for n in pipeline.nodes if n.name == "train_optimized_model")
    assert "X_train" in second_node.inputs
    assert "y_train" in second_node.inputs
    assert "optimized_hyperparameters" in second_node.inputs
    assert "trained_model" in second_node.outputs