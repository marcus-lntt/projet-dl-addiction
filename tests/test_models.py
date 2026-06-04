import numpy as np
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models_dl import MLP, predict_mlp
from src.models_advanced import TemporalTransformer, PositionalEncoding
from src.evaluation import regression_metrics


# ── MLP ───────────────────────────────────────────────────────────────────────

def test_mlp_output_shape():
    model = MLP(input_dim=18, hidden_dims=(64, 32))
    x = torch.randn(16, 18)
    out = model(x)
    assert out.shape == (16,), f"Expected (16,), got {out.shape}"


def test_mlp_predict_numpy():
    model = MLP(input_dim=18, hidden_dims=(32,))
    X = np.random.randn(50, 18).astype(np.float32)
    preds = predict_mlp(model, X)
    assert preds.shape == (50,)
    assert not np.any(np.isnan(preds))


# ── Transformer ───────────────────────────────────────────────────────────────

def test_positional_encoding_shape():
    pe = PositionalEncoding(d_model=64)
    x = torch.randn(4, 10, 64)
    out = pe(x)
    assert out.shape == (4, 10, 64)


def test_transformer_output_shape():
    model = TemporalTransformer(input_size=1, d_model=32, nhead=2, num_encoder_layers=1)
    x = torch.randn(8, 10, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8,), f"Expected (8,), got {out.shape}"


def test_transformer_no_nan():
    model = TemporalTransformer(input_size=1, d_model=32, nhead=2, num_encoder_layers=1)
    x = torch.randn(4, 8, 1)
    with torch.no_grad():
        out = model(x)
    assert not torch.any(torch.isnan(out))


# ── Evaluation ────────────────────────────────────────────────────────────────

def test_regression_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = regression_metrics(y, y, "perfect")
    assert result["R²"] == 1.0
    assert result["RMSE"] == 0.0


def test_regression_metrics_keys():
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)
    result = regression_metrics(y_true, y_pred, "test")
    assert set(result.keys()) == {"model", "R²", "RMSE", "MAE"}
