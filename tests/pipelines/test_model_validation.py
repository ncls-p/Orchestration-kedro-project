import pytest
from kedro.pipeline import Pipeline

from kedro_project.pipelines.model_validation import create_pipeline


@pytest.mark.pipeline("model_validation")
def test_model_validation_pipeline():
    """Test that model validation pipeline is created correctly."""
    pipeline = create_pipeline()
    
    # Check that pipeline is created
    assert isinstance(pipeline, Pipeline)
    
    # Check that pipeline has the expected number of nodes
    assert len(pipeline.nodes) == 7
    
    # Check that all expected nodes are present
    node_names = {node.name for node in pipeline.nodes}
    expected_nodes = {
        "generate_probability_predictions",
        "create_density_chart",
        "create_calibration_curve",
        "create_roc_curve",
        "create_pr_curve",
        "run_invariance_test",
        "run_prototype_test",
    }
    assert node_names == expected_nodes
    
    # Check pipeline inputs and outputs
    assert "trained_model" in pipeline.inputs()
    assert "X_test" in pipeline.inputs()
    assert "y_test" in pipeline.inputs()
    
    # Check visualization outputs
    assert "density_chart" in pipeline.outputs()
    assert "calibration_curve" in pipeline.outputs()
    assert "roc_curve" in pipeline.outputs()
    assert "pr_curve" in pipeline.outputs()
    
    # Check test results outputs
    assert "invariance_test_results" in pipeline.outputs()
    assert "prototype_test_results" in pipeline.outputs()
    # Note: validation_predictions is an intermediate output, not exposed
    
    # Check that first node generates predictions
    first_node = next(n for n in pipeline.nodes if n.name == "generate_probability_predictions")
    assert set(first_node.inputs) == {"trained_model", "X_test", "y_test"}
    assert "validation_predictions" in first_node.outputs
    
    # Check that visualization nodes depend on predictions
    viz_nodes = ["create_density_chart", "create_calibration_curve", "create_roc_curve", "create_pr_curve"]
    for node_name in viz_nodes:
        node = next(n for n in pipeline.nodes if n.name == node_name)
        assert "validation_predictions" in node.inputs