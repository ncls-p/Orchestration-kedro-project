"""
Data processing pipeline nodes for network intrusion detection.

This module contains functions for loading, preprocessing, and transforming
network intrusion detection data for machine learning pipelines.
"""

from typing import Any, cast

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------
# Module-level constants (replaces magic numbers to silence PLR2004)
# --------------------------------------------------------------------
WEEKEND_START_DAY: int = 5
BUSINESS_DAY_START: int = 9
BUSINESS_DAY_END: int = 17

# Export all public functions for backward compatibility
__all__ = [
    "load_and_preprocess_data",
    "create_time_features",
    "create_rolling_features",
    "create_lag_features",
    "create_time_since_features",
    "encode_categorical_features",
    "create_feature_matrix",
    "split_time_series_data",
]


def load_and_preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Load raw data and perform basic preprocessing.

    This function loads the raw network intrusion detection data and performs
    essential preprocessing steps including timestamp conversion and chronological
    sorting to ensure proper time series order.

    Args:
        raw_data: Raw network intrusion detection data containing timestamp
            and other network features

    Returns:
        Preprocessed DataFrame with converted timestamps and chronological ordering

    Examples:
        >>> raw_df = pd.DataFrame(
        ...     {
        ...         "Timestamp": ["2023-01-01 10:00:00", "2023-01-01 09:00:00"],
        ...         "Source_IP": ["192.168.1.1", "192.168.1.2"],
        ...     }
        ... )
        >>> processed = load_and_preprocess_data(raw_df)
        >>> processed["Timestamp"].dtype
        dtype('<M8[ns]')
    """
    data = raw_data.copy()

    # Convert timestamp to datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    # Sort by timestamp to ensure proper time series order
    data = data.sort_values("Timestamp").reset_index(drop=True)

    return data


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from timestamp column.

    This function extracts various time-based features from the timestamp column
    to help capture temporal patterns in network intrusion detection data.

    Args:
        data: DataFrame containing 'Timestamp' column

    Returns:
        DataFrame with additional time-based features including:
            - hour: Hour of the day (0-23)
            - day_of_week: Day of week (0=Monday, 6=Sunday)
            - day: Day of month (1-31)
            - month: Month of year (1-12)
            - minute: Minute of hour (0-59)
            - is_weekend: Binary indicator for weekend (1=weekend, 0=weekday)
            - is_business_hours: Binary indicator for business hours (1=9-17, 0=otherwise)

    Examples:
        >>> df = pd.DataFrame({"Timestamp": [pd.Timestamp("2023-01-01 15:30:00")]})
        >>> result = create_time_features(df)
        >>> result["is_business_hours"].iloc[0]
        1
    """
    data = data.copy()

    # Feature engineering for time series
    data["hour"] = data["Timestamp"].dt.hour
    data["day_of_week"] = data["Timestamp"].dt.dayofweek
    data["day"] = data["Timestamp"].dt.day
    data["month"] = data["Timestamp"].dt.month
    data["minute"] = data["Timestamp"].dt.minute
    data["is_weekend"] = (data["day_of_week"] >= WEEKEND_START_DAY).astype(int)
    data["is_business_hours"] = (
        (data["hour"] >= BUSINESS_DAY_START) & (data["hour"] <= BUSINESS_DAY_END)
    ).astype(int)

    return data


def create_rolling_features(data: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
    """
    Create rolling window features for time series analysis.

    This function creates rolling window statistics from the payload size
    to capture temporal patterns and trends in network traffic.

    Args:
        data: DataFrame containing 'Payload_Size' column
        window_size: Size of the rolling window (default: 10)

    Returns:
        DataFrame with additional rolling window features:
            - payload_rolling_mean: Rolling mean of payload size
            - payload_rolling_std: Rolling standard deviation of payload size

    Examples:
        >>> df = pd.DataFrame({"Payload_Size": [100, 200, 150, 300, 250]})
        >>> result = create_rolling_features(df, window_size=3)
        >>> "payload_rolling_mean" in result.columns
        True
    """
    data = data.copy()

    # Create rolling window features
    data["payload_rolling_mean"] = (
        data["Payload_Size"].rolling(window=window_size, min_periods=1).mean()
    )
    data["payload_rolling_std"] = (
        data["Payload_Size"].rolling(window=window_size, min_periods=1).std().fillna(0)
    )

    return data


def create_lag_features(data: pd.DataFrame, lag_values: list[int]) -> pd.DataFrame:
    """
    Create lag features for time series analysis.

    This function creates lag features from the payload size to capture
    temporal dependencies in network traffic patterns.

    Args:
        data: DataFrame containing 'Payload_Size' column
        lag_values: List of lag periods to create features for

    Returns:
        DataFrame with additional lag features:
            - payload_lag_N: Payload size N periods ago (for each N in lag_values)

    Examples:
        >>> df = pd.DataFrame({"Payload_Size": [100, 200, 150, 300, 250]})
        >>> result = create_lag_features(df, [1, 2])
        >>> "payload_lag_1" in result.columns
        True
    """
    data = data.copy()

    # Create lag features
    for lag in lag_values:
        data[f"payload_lag_{lag}"] = (
            data["Payload_Size"].shift(lag).fillna(data["Payload_Size"].mean())
        )

    return data


def create_time_since_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create time since last similar request features.

    This function calculates the time elapsed since the last similar request
    based on request type, which can help identify patterns in network behavior.

    Args:
        data: DataFrame containing 'Request_Type' and 'Timestamp' columns

    Returns:
        DataFrame with additional time-based feature:
            - time_since_last: Time in seconds since last similar request

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "Request_Type": ["GET", "POST", "GET"],
        ...         "Timestamp": pd.date_range("2023-01-01", periods=3, freq="1H"),
        ...     }
        ... )
        >>> result = create_time_since_features(df)
        >>> "time_since_last" in result.columns
        True
    """
    data = data.copy()

    # Time since last similar request (by Request_Type)
    data["time_since_last"] = (
        data.groupby("Request_Type")["Timestamp"].diff().dt.total_seconds().fillna(0)
    )

    return data


def encode_categorical_features(
    data: pd.DataFrame, categorical_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Encode categorical variables and return encoders.

    This function applies label encoding to categorical columns, creating
    encoded versions while preserving the original encoders for future use.

    Args:
        data: DataFrame containing categorical columns to encode
        categorical_columns: List of column names to encode

    Returns:
        Tuple containing:
            - DataFrame with additional encoded columns (original_name + "_encoded")
            - Dictionary mapping column names to their fitted LabelEncoder objects

    Examples:
        >>> df = pd.DataFrame(
        ...     {"protocol": ["HTTP", "HTTPS", "HTTP"], "status": ["200", "404", "200"]}
        ... )
        >>> result_df, encoders = encode_categorical_features(df, ["protocol"])
        >>> "protocol_encoded" in result_df.columns
        True
    """
    data = data.copy()
    encoders = {}

    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        data[col + "_encoded"] = encoders[col].fit_transform(data[col])

    return data, encoders


def create_feature_matrix(
    data: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create feature matrix X and target y for machine learning.

    This function extracts the feature matrix and target variable from
    the processed data for use in machine learning models.

    Args:
        data: DataFrame containing features and target variable
        feature_columns: List of column names to use as features

    Returns:
        Tuple containing:
            - X: Feature matrix (DataFrame) with selected columns
            - y: Target variable (Series) containing intrusion labels

    Examples:
        >>> df = pd.DataFrame(
        ...     {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "Intrusion": [0, 1, 0]}
        ... )
        >>> X, y = create_feature_matrix(df, ["feature1", "feature2"])
        >>> X.shape
        (3, 2)
    """
    X = cast(pd.DataFrame, data[feature_columns].copy())
    y = cast(pd.Series, data["Intrusion"].copy())

    return X, y


def split_time_series_data(
    X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically for time series analysis.

    This function splits the feature matrix and target variable chronologically
    to preserve temporal order, which is crucial for time series modeling.

    Args:
        X: Feature matrix (DataFrame) to split
        y: Target variable (Series) to split
        train_ratio: Proportion of data to use for training (default: 0.8)

    Returns:
        Tuple containing:
            - X_train: Training feature matrix
            - X_test: Testing feature matrix
            - y_train: Training target variable
            - y_test: Testing target variable

    Examples:
        >>> X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        >>> y = pd.Series([0, 1, 0, 1, 0])
        >>> X_train, X_test, y_train, y_test = split_time_series_data(X, y, 0.6)
        >>> len(X_train)
        3
    """
    split_index = int(len(X) * train_ratio)

    X_train = cast(pd.DataFrame, X[:split_index].copy())
    X_test = cast(pd.DataFrame, X[split_index:].copy())
    y_train = cast(pd.Series, y[:split_index].copy())
    y_test = cast(pd.Series, y[split_index:].copy())

    return X_train, X_test, y_train, y_test
