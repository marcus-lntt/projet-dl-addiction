"""Transformer temporel pour la prévision de séries temporelles d'addiction (Jalon 8).

Justification du choix Transformer vs autres architectures avancées :
- GAN : génératif, adapté à la création de données synthétiques — hors-sujet ici.
- VAE : encodeur-décodeur latent — utile pour l'anomalie detection, pas pour la prévision.
- GNN : graphes — pas de structure de graphe dans ce dataset.
- Transfer Learning : nécessite un modèle pré-entraîné sur une tâche similaire — peu disponible.
- LSTM : précédent état de l'art pour les séries temporelles, capte la mémoire long-terme.
- **Transformer** : état de l'art actuel pour les séries temporelles.
  * L'attention multi-têtes capte les dépendances à TOUTES les distances temporelles simultanément,
    là où le LSTM traite la séquence de façon récurrente (coût quadratique vs linéaire).
  * Parallélisable sur GPU.
  * L'encodage positionnel injecte l'ordre temporel sans récurrence.
  * Confirmé meilleur que LSTM sur de nombreux benchmarks (Informer, PatchTST, etc.).
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─── Positional Encoding ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Injecte l'ordre temporel via des sinusoïdes de fréquences différentes (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─── Temporal Transformer ────────────────────────────────────────────────────

class TemporalTransformer(nn.Module):
    """Transformer encoder-only pour la régression sur séries temporelles univariées.

    Architecture :
        input projection → positional encoding → N × TransformerEncoderLayer → mean pooling → FC
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm : plus stable (gradient flow amélioré)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        x = self.input_proj(x)          # → (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)             # attention multi-têtes sur toute la séquence
        x = x.mean(dim=1)              # mean pooling → (batch, d_model)
        return self.head(x).squeeze(-1)


# ─── Data utilities ───────────────────────────────────────────────────────────

def build_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    def _t(a):
        return torch.tensor(a, dtype=torch.float32)

    train_ds = TensorDataset(_t(X_train).unsqueeze(-1), _t(y_train))
    val_ds = TensorDataset(_t(X_val).unsqueeze(-1), _t(y_val))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_transformer(model, train_loader, val_loader, epochs=200, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            # Gradient clipping — contrer l'exploding gradient dans le Transformer
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_losses = [criterion(model(xb.to(device)), yb.to(device)).item() for xb, yb in val_loader]

        history["train_loss"].append(np.mean(losses))
        history["val_loss"].append(np.mean(val_losses))

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} — Train: {history['train_loss'][-1]:.4f} | Val: {history['val_loss'][-1]:.4f}")

    return model.to("cpu"), history


# ─── Inference ────────────────────────────────────────────────────────────────

def forecast_future(model, last_sequence, n_steps=10):
    """Prévision récursive : chaque prédiction sert d'entrée pour le pas suivant."""
    model.eval()
    seq = list(last_sequence.copy())
    sl = len(last_sequence)
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor(seq[-sl:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            p = model(x).item()
            preds.append(p)
            seq.append(p)
    return np.array(preds)


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_forecast(years_known, values_known, years_future, values_forecast, country=""):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(years_known, values_known, "o-", markersize=3, label="Données observées (2015–2060)")
    ax.plot(years_future, values_forecast, "s--", color="crimson", label="Prévision Transformer")
    ax.axvline(years_known[-1], color="gray", linestyle=":", alpha=0.7, label="Frontière connue/prédit")
    ax.set_title(f"Prévision Transformer — Addiction Score ({country})")
    ax.set_xlabel("Année")
    ax.set_ylabel("Addiction Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
