from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Load raw data and perform basic preprocessing."""
    data = raw_data.copy()

    # Convert timestamp to datetime
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])

    # Sort by timestamp to ensure proper time series order
    data = data.sort_values("Timestamp").reset_index(drop=True)

    return data


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from timestamp."""
    data = data.copy()

    # Feature engineering for time series
    data["hour"] = data["Timestamp"].dt.hour
    data["day_of_week"] = data["Timestamp"].dt.dayofweek
    data["day"] = data["Timestamp"].dt.day
    data["month"] = data["Timestamp"].dt.month
    data["minute"] = data["Timestamp"].dt.minute
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_business_hours"] = ((data["hour"] >= 9) & (data["hour"] <= 17)).astype(int)

    return data


def create_rolling_features(data: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
    """Create rolling window features."""
    data = data.copy()

    # Create rolling window features
    data["payload_rolling_mean"] = (
        data["Payload_Size"].rolling(window=window_size, min_periods=1).mean()
    )
    data["payload_rolling_std"] = (
        data["Payload_Size"].rolling(window=window_size, min_periods=1).std().fillna(0)
    )

    return data


def create_lag_features(data: pd.DataFrame, lag_values: list) -> pd.DataFrame:
    """Create lag features for time series."""
    data = data.copy()

    # Create lag features
    for lag in lag_values:
        data[f"payload_lag_{lag}"] = (
            data["Payload_Size"].shift(lag).fillna(data["Payload_Size"].mean())
        )

    return data


def create_time_since_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create time since last similar request features."""
    data = data.copy()

    # Time since last similar request (by Request_Type)
    data["time_since_last"] = (
        data.groupby("Request_Type")["Timestamp"].diff().dt.total_seconds().fillna(0)
    )

    return data


def encode_categorical_features(data: pd.DataFrame, categorical_columns: list) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode categorical variables and return encoders."""
    data = data.copy()
    encoders = {}

    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        data[col + "_encoded"] = encoders[col].fit_transform(data[col])

    return data, encoders


def create_feature_matrix(data: pd.DataFrame, feature_columns: list) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix X and target y."""
    X = data[feature_columns].copy()
    y = data["Intrusion"].copy()

    return X, y


def split_time_series_data(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically for time series."""
    split_index = int(len(X) * train_ratio)

    X_train = X[:split_index].copy()
    X_test = X[split_index:].copy()
    y_train = y[:split_index].copy()
    y_test = y[split_index:].copy()

    return X_train, X_test, y_train, y_test
