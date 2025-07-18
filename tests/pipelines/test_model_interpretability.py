import pytest
from kedro.pipeline import Pipeline

from kedro_project.pipelines.model_interpretability import create_pipeline


@pytest.mark.pipeline("model_interpretability")
def test_model_interpretability_pipeline():
    """Test that model interpretability pipeline is created correctly."""
    pipeline = create_pipeline()
    
    # Check that pipeline is created
    assert isinstance(pipeline, Pipeline)
    
    # Check that pipeline has the expected number of nodes
    assert len(pipeline.nodes) == 6
    
    # Check that all expected nodes are present
    node_names = {node.name for node in pipeline.nodes}
    expected_nodes = {
        "compute_shap_values",
        "create_shap_summary_plot",
        "create_shap_feature_importance",
        "create_shap_dependence_plots",
        "analyze_local_interpretations",
        "create_global_feature_importance_comparison",
    }
    assert node_names == expected_nodes
    
    # Check pipeline inputs and outputs
    assert "trained_model" in pipeline.inputs()
    assert "X_test" in pipeline.inputs()
    assert "validation_predictions" in pipeline.inputs()
    assert "feature_importance_data" in pipeline.inputs()
    
    # Check SHAP-related outputs (only final outputs are exposed)
    assert "shap_summary_plot" in pipeline.outputs()
    assert "shap_feature_importance_plot" in pipeline.outputs()
    assert "shap_dependence_plots" in pipeline.outputs()
    assert "local_interpretations" in pipeline.outputs()
    assert "feature_importance_comparison_plot" in pipeline.outputs()
    # Note: shap_data and shap_feature_importance are intermediate outputs, not exposed
    
    # Check that first node computes SHAP values
    first_node = next(n for n in pipeline.nodes if n.name == "compute_shap_values")
    assert set(first_node.inputs) == {"trained_model", "X_test"}
    assert "shap_data" in first_node.outputs
    
    # Check that SHAP visualization nodes depend on shap_data
    shap_viz_nodes = ["create_shap_summary_plot", "create_shap_feature_importance", "create_shap_dependence_plots"]
    for node_name in shap_viz_nodes:
        node = next(n for n in pipeline.nodes if n.name == node_name)
        assert "shap_data" in node.inputs
    
    # Check that local interpretations node has correct inputs
    local_node = next(n for n in pipeline.nodes if n.name == "analyze_local_interpretations")
    assert "shap_data" in local_node.inputs
    assert "validation_predictions" in local_node.inputs
    assert "params:n_local_samples" in local_node.inputs