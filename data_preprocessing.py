"""
InteliCrop - Data Preprocessing Module
Handles data loading, cleaning, feature engineering, and train/test splitting.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the crop recommendation dataset from CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to CSV file. Defaults to data/Crop_recommendation.csv

    Returns
    -------
    pd.DataFrame
        Loaded dataset

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "Crop_recommendation.csv",
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Run `python -m src.generate_dataset` first or place the CSV in data/."
        )

    df = pd.read_csv(filepath)
    print(f"[OK] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def get_feature_target(df: pd.DataFrame):
    """
    Split dataframe into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'label' column

    Returns
    -------
    tuple
        (X, y) where X is feature DataFrame and y is target Series
    """
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[feature_cols]
    y = df["label"]
    return X, y


def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Complete preprocessing pipeline.
    
    Steps:
    1. Handle missing values (forward fill + median)
    2. Encode target labels
    3. Scale features
    4. Train/test split

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing X_train, X_test, y_train, y_test,
        scaler, label_encoder, and feature_names
    """
    # --- 1. Handle missing values ---
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # --- 2. Separate features and target ---
    X, y = get_feature_target(df)
    feature_names = list(X.columns)

    # --- 3. Encode target labels ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- 4. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    # --- 5. Feature scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[OK] Preprocessing complete")
    print(f"    Train: {X_train_scaled.shape[0]} samples | Test: {X_test_scaled.shape[0]} samples")
    print(f"    Features: {feature_names}")
    print(f"    Classes: {list(le.classes_)}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoder": le,
        "feature_names": feature_names,
    }


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a statistical summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to summarize

    Returns
    -------
    dict
        Summary statistics
    """
    X, y = get_feature_target(df)
    return {
        "shape": df.shape,
        "features": list(X.columns),
        "num_classes": y.nunique(),
        "class_names": sorted(y.unique().tolist()),
        "class_distribution": y.value_counts().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "statistics": X.describe().to_dict(),
    }
