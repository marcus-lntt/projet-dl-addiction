# Livrables — Checklist de rendu (prof)

Ce document sert de "mode d’emploi" pour corriger / reproduire le projet rapidement.

## 1) Ce que contient le dépôt

- Code métier : `src/`
- Tests : `tests/`
- Notebooks : `notebooks/` (committés avec sorties déjà exécutées)
- Dashboard : `app/streamlit_app.py`
- CI : `.github/workflows/ci.yml`

Ce que le dépôt ne contient pas :

- Le dataset (`data/`) : volontairement non versionné (voir `.gitignore`)
- Les artefacts de modèles (`models/`) : non versionnés (voir `.gitignore`)

## 2) Installation (reproductibilité)

Prérequis : Python >= 3.10 et `uv`.

```bash
uv sync
```

> Astuce : la CI utilise le lockfile `uv.lock` via `uv sync --locked`.

## 3) Vérification rapide (tests)

```bash
uv run pytest -v
```

Attendu : tous les tests passent.

## 4) Exécuter les notebooks (ordre)

> Les notebooks sont déjà commit avec sorties exécutées.
> Pour re-exécuter localement :

```bash
uv run jupyter notebook notebooks/
```

Ordre :

1. `01_eda.ipynb` — chargement & EDA + contrôle fuite de données
2. `02_ml_baseline.ipynb` — Ridge/Lasso/ElasticNet + métriques
3. `03_dl_fondamental.ipynb` — MLP + entraînement + optimisation
4. `04_dl_avance.ipynb` — Transformer temporel

## 5) Lancer le dashboard Streamlit

```bash
uv run streamlit run app/streamlit_app.py
```

Points à vérifier (correction) :

- la page d’accueil se charge
- la page de prédiction produit un score
- les visualisations (Plotly) s’affichent

## 6) Ce que je conseille de montrer en soutenance / rendu

- Une exécution de `pytest` (preuve qualité)
- Un screenshot du dashboard Streamlit
- Dans `01_eda` : justification dataset + démonstration de la fuite via `ASI`
- Dans `02_ml_baseline` : baseline et interprétation des métriques
- Dans `03_dl_fondamental` : justification MLP + courbes d’entraînement
- Dans `04_dl_avance` : description du transformer temporel (hypothèses/limites)

## 7) Problèmes fréquents

Voir `docs/TROUBLESHOOTING.md`.
