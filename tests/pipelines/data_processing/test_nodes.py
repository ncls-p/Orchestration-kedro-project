"""Tests for data processing pipeline nodes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from kedro_project.pipelines.data_processing.nodes import (
    create_feature_matrix,
    create_lag_features,
    create_rolling_features,
    create_time_features,
    create_time_since_features,
    encode_categorical_features,
    load_and_preprocess_data,
    split_time_series_data,
)


class TestDataProcessingNodes:
    """Test suite for data processing pipeline nodes."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        return pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(
                    [
                        "2025-01-01 10:00:00",
                        "2025-01-01 11:00:00",
                        "2025-01-01 12:00:00",
                    ]
                ),
                "Source_IP": ["192.168.1.1", "192.168.1.2", "192.168.1.3"],
                "Destination_IP": ["10.0.0.1", "10.0.0.2", "10.0.0.3"],
                "Port": [80, 443, 8080],
                "Request_Type": ["GET", "POST", "GET"],
                "Protocol": ["HTTP", "HTTPS", "HTTP"],
                "Payload_Size": [1024, 2048, 1536],
                "User_Agent": ["Mozilla/5.0", "Chrome/91.0", "Safari/14.0"],
                "Status": ["Success", "Error", "Success"],
                "Intrusion": [0, 1, 0],
                "Scan_Type": ["Normal", "Attack", "Normal"],
            }
        )

    def test_load_and_preprocess_data(self, sample_raw_data):
        """Test load_and_preprocess_data function."""
        result = load_and_preprocess_data(sample_raw_data)

        # Check that timestamp is converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(result["Timestamp"])

        # Check that data is sorted by timestamp
        assert result["Timestamp"].is_monotonic_increasing

        # Check that all original columns are preserved
        assert all(col in result.columns for col in sample_raw_data.columns)

    def test_create_time_features(self, sample_raw_data):
        """Test create_time_features function."""
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        result = create_time_features(preprocessed_data)

        # Check that time features are created
        expected_features = [
            "hour",
            "day_of_week",
            "day",
            "month",
            "minute",
            "is_weekend",
            "is_business_hours",
        ]
        assert all(feature in result.columns for feature in expected_features)

        # Check hour extraction
        assert result["hour"].iloc[0] == 10
        assert result["hour"].iloc[1] == 11
        assert result["hour"].iloc[2] == 12

        # Check business hours logic
        assert result["is_business_hours"].iloc[0] == 1  # 10:00 is business hour
        assert result["is_business_hours"].iloc[1] == 1  # 11:00 is business hour
        assert result["is_business_hours"].iloc[2] == 1  # 12:00 is business hour

    def test_create_rolling_features(self, sample_raw_data):
        """Test create_rolling_features function."""
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        time_features = create_time_features(preprocessed_data)
        result = create_rolling_features(time_features, window_size=2)

        # Check that rolling features are created
        assert "payload_rolling_mean" in result.columns
        assert "payload_rolling_std" in result.columns

        # Check rolling mean calculation
        assert result["payload_rolling_mean"].iloc[0] == 1024.0  # First value
        assert (
            result["payload_rolling_mean"].iloc[1] == (1024 + 2048) / 2
        )  # Mean of first two

    def test_create_lag_features(self, sample_raw_data):
        """Test create_lag_features function."""
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        time_features = create_time_features(preprocessed_data)
        rolling_features = create_rolling_features(time_features)
        result = create_lag_features(rolling_features, lag_values=[1, 2])

        # Check that lag features are created
        assert "payload_lag_1" in result.columns
        assert "payload_lag_2" in result.columns

        # Check lag calculation
        assert (
            result["payload_lag_1"].iloc[1] == 1024
        )  # Second row should have first row's value
        assert (
            result["payload_lag_1"].iloc[2] == 2048
        )  # Third row should have second row's value

    def test_create_time_since_features(self, sample_raw_data):
        """Test create_time_since_features function."""
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        time_features = create_time_features(preprocessed_data)
        rolling_features = create_rolling_features(time_features)
        lag_features = create_lag_features(rolling_features, lag_values=[1])
        result = create_time_since_features(lag_features)

        # Check that time since feature is created
        assert "time_since_last" in result.columns

        # First row should have 0 (no previous request)
        assert result["time_since_last"].iloc[0] == 0

        # Check that time differences are calculated
        # Row 1 is first POST request, so should be 0
        assert result["time_since_last"].iloc[1] == 0  # First POST request
        # Row 2 is second GET request, so should be > 0
        assert result["time_since_last"].iloc[2] > 0  # Should be 2 hours (7200 seconds)

    def test_encode_categorical_features(self, sample_raw_data):
        """Test encode_categorical_features function."""
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        time_features = create_time_features(preprocessed_data)
        rolling_features = create_rolling_features(time_features)
        lag_features = create_lag_features(rolling_features, lag_values=[1])
        time_since_features = create_time_since_features(lag_features)

        categorical_columns = ["Request_Type", "Protocol", "User_Agent", "Status"]
        result, encoders = encode_categorical_features(
            time_since_features, categorical_columns
        )

        # Check that encoded features are created
        for col in categorical_columns:
            assert f"{col}_encoded" in result.columns
            assert col in encoders
            assert isinstance(encoders[col], LabelEncoder)

        # Check encoding values
        assert (
            result["Request_Type_encoded"].iloc[0]
            == result["Request_Type_encoded"].iloc[2]
        )  # Both 'GET'
        assert (
            result["Request_Type_encoded"].iloc[1]
            != result["Request_Type_encoded"].iloc[0]
        )  # 'POST' vs 'GET'

    def test_create_feature_matrix(self, sample_raw_data):
        """Test create_feature_matrix function."""
        # Process data through the pipeline
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        time_features = create_time_features(preprocessed_data)
        rolling_features = create_rolling_features(time_features)
        lag_features = create_lag_features(rolling_features, lag_values=[1])
        time_since_features = create_time_since_features(lag_features)
        categorical_columns = ["Request_Type", "Protocol", "User_Agent", "Status"]
        encoded_features, _ = encode_categorical_features(
            time_since_features, categorical_columns
        )

        # Define feature columns
        feature_columns = [
            "Payload_Size",
            "hour",
            "day_of_week",
            "day",
            "month",
            "minute",
            "is_weekend",
            "is_business_hours",
            "payload_rolling_mean",
            "payload_rolling_std",
            "payload_lag_1",
            "time_since_last",
        ] + [col + "_encoded" for col in categorical_columns]

        X, y = create_feature_matrix(encoded_features, feature_columns)

        # Check that feature matrix has correct shape
        assert X.shape[0] == len(sample_raw_data)
        assert X.shape[1] == len(feature_columns)

        # Check that target vector is correct
        assert len(y) == len(sample_raw_data)
        assert list(y) == [0, 1, 0]  # From sample data

    def test_split_time_series_data(self, sample_raw_data):
        """Test split_time_series_data function."""
        # Create sample feature matrix and target
        X = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]}
        )
        y = pd.Series([0, 1, 0, 1, 0])

        X_train, X_test, y_train, y_test = split_time_series_data(X, y, train_ratio=0.6)

        # Check split sizes
        assert len(X_train) == 3  # 60% of 5
        assert len(X_test) == 2  # 40% of 5
        assert len(y_train) == 3
        assert len(y_test) == 2

        # Check that split maintains chronological order
        assert X_train.iloc[0]["feature1"] == 1
        assert X_train.iloc[-1]["feature1"] == 3
        assert X_test.iloc[0]["feature1"] == 4
        assert X_test.iloc[-1]["feature1"] == 5

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()

        # Should handle empty dataframe gracefully
        with pytest.raises((ValueError, KeyError)):
            load_and_preprocess_data(empty_df)

        # Test with single row
        single_row = pd.DataFrame(
            {
                "Timestamp": ["2025-01-01 10:00:00"],
                "Payload_Size": [1024],
                "Request_Type": ["GET"],
                "Intrusion": [0],
            }
        )

        result = load_and_preprocess_data(single_row)
        assert len(result) == 1
        assert pd.api.types.is_datetime64_any_dtype(result["Timestamp"])

    def test_data_types(self, sample_raw_data):
        """Test that output data types are correct."""
        preprocessed_data = load_and_preprocess_data(sample_raw_data)
        time_features = create_time_features(preprocessed_data)

        # Check that time features have correct data types (int32 or int64)
        assert time_features["hour"].dtype in [np.int32, np.int64]
        assert time_features["is_weekend"].dtype in [np.int32, np.int64]
        assert time_features["is_business_hours"].dtype in [np.int32, np.int64]

        # Check that rolling features have correct data types
        rolling_features = create_rolling_features(time_features)
        assert rolling_features["payload_rolling_mean"].dtype == np.float64
        assert rolling_features["payload_rolling_std"].dtype == np.float64
