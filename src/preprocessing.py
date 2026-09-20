"""Preprocessing utilities: loading, cleaning, encoding, splitting, scaling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from . import FEATURES

RANDOM_STATE = 42


@dataclass
class PreprocessResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    label_encoder: LabelEncoder


REQUIRED_COLUMNS = FEATURES + ["label"]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the crop dataset and validate its schema."""
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[list(FEATURES)].isnull().any().any():
        raise ValueError("Dataset contains missing feature values")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2) -> PreprocessResult:
    """Encode labels, stratify-split, and scale features (fit on train only)."""
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df["label"]), name="label")
    X = df[FEATURES].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=FEATURES, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

    return PreprocessResult(X_train, X_test, y_train, y_test, scaler, label_encoder)
