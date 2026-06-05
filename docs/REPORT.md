# Rapport synthèse — Projet Deep Learning (APTI)

**Auteur : Marcus LINGUET**  
**Repo :** https://github.com/marcus-lntt/projet-dl-addiction

## Jalon 1 — Data (choix et justification)

### Objectif
Prédire un score continu `addiction_score` (0–100) à partir de variables socio‑démographiques et d’usage (TikTok/Instagram), avec une dimension temporelle (années 2015–2060).

### Pourquoi ce dataset est pertinent
- **Régression tabulaire** : les variables sont numériques/catégorielles, adaptées à une baseline ML (Ridge/Lasso/ElasticNet) et à un MLP.
- **Dimension temporelle** : permet d’introduire ensuite une approche séquentielle (Transformer temporel) au Jalon 8.
- **Taille** : suffisamment grande pour un entraînement DL, tout en restant exécutable sur machine personnelle.

### Qualité & précaution : fuite de données
La variable `ASI` est exclue des features car elle est fortement corrélée à la cible (composite qui encode indirectement `addiction_score`). Elle n’est pas utilisée dans l’entraînement (`FEATURES` dans `src/preprocessing.py`).

## Reproductibilité & qualité logicielle

- Environnement : `uv` + `uv.lock` (installation stable)
- Tests : `pytest` (préprocessing + shapes modèles)
- CI : GitHub Actions (installation + tests automatisés)
