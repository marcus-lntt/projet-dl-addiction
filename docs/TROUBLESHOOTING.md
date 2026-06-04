# Troubleshooting

## `uv` introuvable

- Vérifier : `command -v uv`
- Réinstaller : `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Le dataset manque

Symptômes : erreurs de lecture CSV dans les notebooks.

- Télécharger les CSV depuis Kaggle et les placer dans `data/`.
- Le dépôt ignore `data/` par design (pas de gros fichiers versionnés).

## Streamlit démarre mais certaines pages échouent

Souvent dû à l’absence d’artefacts de modèles (puisque `models/` n’est pas versionné).

- Exécuter les notebooks 02/03 pour régénérer les modèles.
- Puis relancer : `uv run streamlit run app/streamlit_app.py`

## Port Streamlit déjà utilisé

- Lancer sur un autre port :

```bash
uv run streamlit run app/streamlit_app.py --server.port 8502
```

## Erreur PyTorch `ReduceLROnPlateau(... verbose=...)`

Certaines versions de PyTorch ne supportent plus `verbose`.

- Le code gère désormais les deux cas (fallback) dans `src/models_dl.py`.

## CI échoue sur GitHub Actions

- Vérifier l’onglet Actions et le log.
- Les erreurs typiques sont :
  - dépendance manquante → corriger `pyproject.toml` + `uv.lock`
  - test trop strict → ajuster le test (sans baisser la qualité)
