import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, model_name="Model") -> dict:
    return {
        "model": model_name,
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """Tableau comparatif trié par R² décroissant."""
    return pd.DataFrame(results).set_index("model").sort_values("R²", ascending=False)


def plot_predictions(y_true, y_pred, model_name="Model"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Prédit vs réel
    axes[0].scatter(y_true, y_pred, alpha=0.35, s=12)
    lim = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    axes[0].plot(lim, lim, "r--", lw=2, label="Prédiction parfaite")
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Valeurs réelles")
    axes[0].set_ylabel("Valeurs prédites")
    axes[0].set_title(f"{model_name} — Prédit vs Réel")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution des résidus
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=40, edgecolor="black", alpha=0.75)
    axes[1].axvline(0, color="red", linestyle="--", label="Résidu = 0")
    axes[1].set_title(f"{model_name} — Distribution des résidus")
    axes[1].set_xlabel("Résidus")
    axes[1].set_ylabel("Fréquence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def bias_variance_analysis(train_r2, val_r2, model_name="Model"):
    """Interprétation textuelle du compromis biais/variance."""
    gap = train_r2 - val_r2
    print(f"\n{'='*50}")
    print(f"Analyse Biais/Variance — {model_name}")
    print(f"  Train R²      : {train_r2:.4f}")
    print(f"  Validation R² : {val_r2:.4f}")
    print(f"  Écart         : {gap:.4f}")
    if val_r2 < 0.5:
        print("  Diagnostic    : SOUS-APPRENTISSAGE (biais élevé) — modèle trop simple")
    elif gap > 0.1:
        print("  Diagnostic    : SUR-APPRENTISSAGE (variance élevée) — régulariser davantage")
    else:
        print("  Diagnostic    : BON ÉQUILIBRE biais/variance")
    print(f"{'='*50}\n")
