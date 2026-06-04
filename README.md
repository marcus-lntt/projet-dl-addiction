# Projet Deep Learning — Social Media Addiction Score Predictor

**Auteur** : Marcus LINGUET  
**Formation** : Deep Learning — APTI  
**Dataset** : TikTok & Instagram Addiction Dataset (2015–2060) — Kaggle (CC BY-NC-SA 4.0)  
**Tâche** : Régression — prédiction du score d'addiction aux réseaux sociaux (0–100)

---

## Structure du projet

```
projet-dl-addiction/
├── .github/workflows/ci.yml       ← CI/CD GitHub Actions (tests automatisés)
├── data/                          ← Dataset (NON versionné — .gitignore)
│   └── .gitkeep
├── notebooks/
│   ├── 01_eda.ipynb               ← Jalons 1 & 2 — Data & EDA
│   ├── 02_ml_baseline.ipynb       ← Jalons 3 & 4 — Ridge / Lasso / ElasticNet
│   ├── 03_dl_fondamental.ipynb    ← Jalons 5, 6 & 7 — MLP, Optuna, ML vs DL
│   └── 04_dl_avance.ipynb         ← Jalon 8 — Temporal Transformer
├── src/                           ← Code métier modulaire (importé dans les notebooks)
│   ├── preprocessing.py           ← Chargement, encodage, séquences
│   ├── models_ml.py               ← Ridge, Lasso, ElasticNet + visualisations
│   ├── models_dl.py               ← Classe MLP + boucle d'entraînement PyTorch
│   ├── models_advanced.py         ← Temporal Transformer (Jalon 8)
│   └── evaluation.py             ← Métriques, comparaison, analyse biais/variance
├── app/
│   └── streamlit_app.py           ← Jalon 9 — Dashboard interactif
├── tests/
│   ├── test_preprocessing.py      ← Tests unitaires preprocessing
│   └── test_models.py            ← Tests unitaires modèles DL
├── models/                        ← Modèles sauvegardés après exécution (non versionnés)
├── pyproject.toml                 ← Dépendances uv
├── requirements.txt               ← Export pip des dépendances
└── README.md
```

---

## Dataset

| Fichier | Lignes | Contenu |
|---|---|---|
| `tiktok_instagram_global_100countries.csv` | 10 000 | Dataset principal — 100 pays, 2015–2060, 23 colonnes |
| `screen_time_behavior.csv` | 50 000 | Comportements screen time par plateforme, genre, âge |
| `country_wise_analysis_addiction.csv` | 100 | Agrégats par pays |

**Le dataset est exclu du dépôt Git** (voir `.gitignore`).  
Télécharger depuis [Kaggle](https://www.kaggle.com/datasets/abdulmaliklodhra/tiktok-and-instagram-addiction-dataset-20152060) et placer les CSV dans `data/`.

**Note sur le data leakage** : la colonne `ASI` (Addiction Score Index) a été exclue des features car elle encode directement la cible `addiction_score` (corrélation 0.95). Son exclusion rend le problème plus réaliste et pédagogiquement honnête (R² passe de 1.0 → 0.74 pour Ridge).

---

## Installation et exécution

### Prérequis
```bash
# Installer uv (gestionnaire de paquets moderne)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Environnement Python
```bash
uv sync          # crée .venv et installe toutes les dépendances
```

### Lancer les notebooks (dans l'ordre)
```bash
uv run jupyter notebook notebooks/
# Exécuter : 01_eda → 02_ml_baseline → 03_dl_fondamental → 04_dl_avance
```

### Lancer le dashboard Streamlit
```bash
# Après avoir exécuté les notebooks 02 et 03 (pour avoir les modèles sauvegardés)
uv run streamlit run app/streamlit_app.py
```

### Lancer les tests
```bash
uv run pytest tests/ -v
```

---

## Démarche scientifique

### NOTE 1 — Data Science & Baseline ML

| Jalon | Notebook | Résumé |
|---|---|---|
| **1 — Data** | `01_eda.ipynb` | Dataset synthétique (2015–2060), tâche de régression sur `addiction_score`, 100 pays |
| **2 — EDA** | `01_eda.ipynb` | Pas de valeurs manquantes, tendance haussière 2015→2060, classes déséquilibrées (80% High), détection leakage ASI |
| **3 — ML Classique** | `02_ml_baseline.ipynb` | Ridge R²=0.74 / Lasso avec sélection de features / ElasticNet, GridSearchCV 5-fold |
| **4 — Éval ML** | `02_ml_baseline.ipynb` | Courbes d'apprentissage, analyse biais/variance, feature importance, résidus |

**Labels Git** : `[data]`, `[eda]`, `[ml]`, `[eval-ml]`

### NOTE 2 — Deep Learning Fondamental

| Jalon | Notebook | Résumé |
|---|---|---|
| **5 — Architecture** | `03_dl_fondamental.ipynb` | MLP justifié pour données tabulaires (vs CNN pour images, RNN pour séquences) |
| **6 — Optimisation** | `03_dl_fondamental.ipynb` | Optuna (20 trials), BatchNorm + gradient clipping anti-vanishing, Adam vs SGD |
| **7 — Comparaison** | `03_dl_fondamental.ipynb` | ML vs DL : métriques, temps d'inférence, limites de chaque approche |

**Labels Git** : `[dl]`, `[opti-dl]`, `[eval-dl]`

### NOTE 3 — DL Avancé & Ingénierie

| Jalon | Fichier | Résumé |
|---|---|---|
| **8 — Transformer** | `04_dl_avance.ipynb` | Temporal Transformer (attention multi-têtes, positional encoding, Pre-LayerNorm) — prévision 2061–2070 par pays |
| **9 — Streamlit** | `app/streamlit_app.py` | 4 pages : accueil, prédiction personnalisée, carte mondiale choroplèthe, tendances temporelles |
| **CI/CD** | `.github/workflows/ci.yml` | GitHub Actions : uv sync + pytest sur push/PR |

---

## Labels Git recommandés

```bash
# Après chaque jalon, committer avec le label correspondant :
git add notebooks/01_eda.ipynb && git commit -m "[data] Jalon 1 : choix et justification du dataset"
git add notebooks/01_eda.ipynb && git commit -m "[eda] Jalon 2 : EDA complète avec détection data leakage"
git add notebooks/02_ml_baseline.ipynb src/ && git commit -m "[ml] Jalon 3 : baseline Ridge/Lasso/ElasticNet"
git add notebooks/02_ml_baseline.ipynb && git commit -m "[eval-ml] Jalon 4 : évaluation et analyse biais/variance"
git add notebooks/03_dl_fondamental.ipynb src/models_dl.py && git commit -m "[dl] Jalon 5 : architecture MLP"
git add notebooks/03_dl_fondamental.ipynb && git commit -m "[opti-dl] Jalon 6 : optimisation Optuna + anti-vanishing"
git add notebooks/03_dl_fondamental.ipynb && git commit -m "[eval-dl] Jalon 7 : comparaison ML vs DL"
```

---

## Transparence IA

Ce projet a été développé avec l'assistance de **Claude Sonnet (Anthropic)** pour :
- L'architecture modulaire du projet (`src/`, notebooks, app Streamlit)
- L'implémentation du Transformer temporel avec positional encoding
- La détection du data leakage sur la colonne `ASI`
- La suite de tests unitaires (`tests/`)
- La configuration CI/CD GitHub Actions

**Limites identifiées avec l'IA :**
- Le dataset est **synthétique** : les corrélations sont artificiellement fortes, les performances ML sont élevées mais peu généralisables à des données réelles.
- La prévision LSTM/Transformer au-delà de 2060 est une extrapolation mathématique, pas une prévision scientifique.
- L'IA ne peut pas exécuter les notebooks ni vérifier les résultats visuels — relecture humaine indispensable.

Tout le code a été relu, compris et validé manuellement avant soumission.
