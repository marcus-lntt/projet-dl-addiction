import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import encode_features, prepare_sequences, FEATURES, TARGET


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "user_id": range(n),
        "country": np.random.choice(["France", "Germany", "Japan"], n),
        "age": np.random.randint(15, 65, n),
        "year": np.random.randint(2015, 2061, n),
        "internet_penetration": np.random.uniform(30, 100, n),
        "mental_health_support_index": np.random.uniform(20, 90, n),
        "gdp_index": np.random.uniform(0.1, 1.0, n),
        "youth_population_ratio": np.random.uniform(0.1, 0.5, n),
        "baseline_addiction_pressure": np.random.uniform(20, 80, n),
        "tiktok_minutes_daily": np.random.uniform(0, 300, n),
        "instagram_minutes_daily": np.random.uniform(0, 300, n),
        "night_usage_ratio": np.random.uniform(0, 1, n),
        "scroll_velocity": np.random.uniform(1, 6, n),
        "addiction_pressure": np.random.uniform(20, 80, n),
        "attention_span_score": np.random.uniform(40, 100, n),
        "dopamine_dependency_score": np.random.uniform(10, 60, n),
        "impulsivity_index": np.random.uniform(20, 80, n),
        "sleep_hours": np.random.uniform(4, 10, n),
        "sleep_quality_index": np.random.uniform(3, 10, n),
        "ASI": np.random.uniform(30, 100, n),
        "MHRI": np.random.uniform(30, 100, n),
        "addiction_score": np.random.uniform(10, 75, n),
        "addiction_level": np.random.choice(["Low", "Medium", "High"], n),
    })


def test_encode_features_adds_column(sample_df):
    df_enc, le = encode_features(sample_df)
    assert "country_enc" in df_enc.columns


def test_encode_features_no_nan(sample_df):
    df_enc, _ = encode_features(sample_df)
    assert df_enc["country_enc"].isnull().sum() == 0


def test_features_list_length():
    assert len(FEATURES) == 18  # ASI exclu (data leakage)


def test_features_no_asi():
    assert "ASI" not in FEATURES, "ASI crée un data leakage — ne doit pas être dans FEATURES"


def test_prepare_sequences_shape():
    values = np.arange(20, dtype=np.float32)
    X, y = prepare_sequences(values, seq_len=5)
    assert X.shape == (15, 5)
    assert y.shape == (15,)


def test_prepare_sequences_values():
    values = np.arange(10, dtype=np.float32)
    X, y = prepare_sequences(values, seq_len=3)
    np.testing.assert_array_equal(X[0], [0, 1, 2])
    assert y[0] == 3.0
