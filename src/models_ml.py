import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, learning_curve


def train_ridge(X_train, y_train, alphas=None):
    if alphas is None:
        alphas = [0.01, 0.1, 1, 10, 100, 1000]
    grid = GridSearchCV(Ridge(), {"alpha": alphas}, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Ridge — meilleur alpha: {grid.best_params_['alpha']} | CV R²: {grid.best_score_:.4f}")
    return grid


def train_lasso(X_train, y_train, alphas=None):
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10]
    grid = GridSearchCV(
        Lasso(max_iter=10000), {"alpha": alphas}, cv=5, scoring="r2", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"Lasso — meilleur alpha: {grid.best_params_['alpha']} | CV R²: {grid.best_score_:.4f}")
    return grid


def train_elasticnet(X_train, y_train):
    param_grid = {
        "alpha": [0.01, 0.1, 1, 10],
        "l1_ratio": [0.2, 0.5, 0.8],
    }
    grid = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(
        f"ElasticNet — meilleurs params: {grid.best_params_} | CV R²: {grid.best_score_:.4f}"
    )
    return grid


def plot_learning_curve(estimator, X, y, title="Courbe d'apprentissage"):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, "o-", label="Train R²")
    ax.plot(train_sizes, val_mean, "s-", label="Validation R²")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Taille du jeu d'entraînement")
    ax.set_ylabel("R²")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_regularization_path(X_train, y_train, feature_names=None, n_alphas=60):
    """Visualise comment les coefficients évoluent avec la force de régularisation."""
    alphas = np.logspace(-3, 3, n_alphas)
    coefs_ridge, coefs_lasso = [], []
    for a in alphas:
        coefs_ridge.append(Ridge(alpha=a).fit(X_train, y_train).coef_)
        coefs_lasso.append(Lasso(alpha=a, max_iter=10000).fit(X_train, y_train).coef_)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(len(coefs_ridge[0])):
        label = feature_names[i] if feature_names is not None else None
        ax1.plot(alphas, [c[i] for c in coefs_ridge], label=label)
        ax2.plot(alphas, [c[i] for c in coefs_lasso])
    for ax, title in zip([ax1, ax2], ["Ridge (L2)", "Lasso (L1) — sélection de features"]):
        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Coefficients")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    if feature_names is not None:
        ax1.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    return fig
