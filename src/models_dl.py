import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Multi-Layer Perceptron pour la régression sur données tabulaires.

    Choix architectural justifié :
    - Données tabulaires (pas de structure spatiale → pas de CNN,
      pas de séquence temporelle par individu → pas de RNN).
    - BatchNorm après chaque couche linéaire : stabilise les activations
      et atténue le problème de vanishing/exploding gradient.
    - Dropout : régularisation pour limiter l'overfitting.
    """

    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    def _ds(X, y):
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    return (
        DataLoader(_ds(X_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(_ds(X_test, y_test), batch_size=batch_size),
    )


def train_mlp(model, train_loader, val_loader, epochs=150, lr=1e-3, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=15,
            factor=0.5,
            verbose=False,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=15,
            factor=0.5,
        )
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())

        tl, vl = np.mean(train_losses), np.mean(val_losses)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step(vl)

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/{epochs} — Train MSE: {tl:.4f} | Val MSE: {vl:.4f}")

    return model.to("cpu"), history


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def plot_training_history(history, title="Courbes d'entraînement MLP"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["train_loss"], label="Train MSE")
    ax.plot(history["val_loss"], label="Val MSE")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
