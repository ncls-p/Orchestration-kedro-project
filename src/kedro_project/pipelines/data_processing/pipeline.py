from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_feature_matrix,
    create_lag_features,
    create_rolling_features,
    create_time_features,
    create_time_since_features,
    encode_categorical_features,
    load_and_preprocess_data,
    split_time_series_data,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_and_preprocess_data,
                inputs="raw_network_logs",
                outputs="preprocessed_data",
                name="load_and_preprocess_data",
            ),
            node(
                func=create_time_features,
                inputs="preprocessed_data",
                outputs="data_with_time_features",
                name="create_time_features",
            ),
            node(
                func=create_rolling_features,
                inputs=["data_with_time_features", "params:rolling_window_size"],
                outputs="data_with_rolling_features",
                name="create_rolling_features",
            ),
            node(
                func=create_lag_features,
                inputs=["data_with_rolling_features", "params:lag_values"],
                outputs="data_with_lag_features",
                name="create_lag_features",
            ),
            node(
                func=create_time_since_features,
                inputs="data_with_lag_features",
                outputs="data_with_time_since_features",
                name="create_time_since_features",
            ),
            node(
                func=encode_categorical_features,
                inputs=["data_with_time_since_features", "params:categorical_columns"],
                outputs=["data_with_encoded_features", "label_encoders"],
                name="encode_categorical_features",
            ),
            node(
                func=create_feature_matrix,
                inputs=["data_with_encoded_features", "params:feature_columns"],
                outputs=["feature_matrix", "target_vector"],
                name="create_feature_matrix",
            ),
            node(
                func=split_time_series_data,
                inputs=["feature_matrix", "target_vector", "params:train_ratio"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_time_series_data",
            ),
        ]
    )
