import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "../data/tiktok_instagram_global_100countries.csv"


# ASI (Addiction Score Index) est exclu : c'est un composite qui encode directement
# le target addiction_score → fuite de données (data leakage).
FEATURES = [
    "age",
    "year",
    "internet_penetration",
    "mental_health_support_index",
    "gdp_index",
    "youth_population_ratio",
    "baseline_addiction_pressure",
    "tiktok_minutes_daily",
    "instagram_minutes_daily",
    "night_usage_ratio",
    "scroll_velocity",
    "attention_span_score",
    "dopamine_dependency_score",
    "impulsivity_index",
    "sleep_hours",
    "sleep_quality_index",
    "MHRI",
    "country_enc",
]
TARGET = "addiction_score"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def encode_features(df):
    df = df.copy()
    le = LabelEncoder()
    df["country_enc"] = le.fit_transform(df["country"])
    return df, le


def prepare_ml(df, test_size=0.2, random_state=42):
    df_enc, le = encode_features(df)
    X = df_enc[FEATURES].values
    y = df_enc[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test, scaler, le


def prepare_sequences(values, seq_len=5):
    """Build (X, y) time-series pairs from a 1-D array of annual values."""
    values = np.array(values, dtype=np.float32)
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def get_country_series(df, country, target=TARGET):
    """Return sorted annual target values for a single country."""
    return (
        df[df["country"] == country]
        .sort_values("year")[target]
        .values.astype(np.float32)
    )
