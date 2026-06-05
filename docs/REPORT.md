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

## Jalon 4 — Évaluation ML (biais / variance)

### Métriques
Les modèles ML sont évalués via des métriques de régression (ex. MSE, MAE, $R^2$) sur un split train/test.

### Biais / variance
L’évaluation inclut une analyse du compromis biais/variance :

- **Sous-apprentissage (biais élevé)** : erreurs élevées en train *et* en test.
- **Sur-apprentissage (variance élevée)** : erreur très basse en train, mais haute en test.

Les courbes d’apprentissage (learning curves) permettent de visualiser ces régimes et d’ajuster la régularisation (`alpha`) et/ou la quantité de données.

### Où retrouver dans le projet
- Notebook : `notebooks/02_ml_baseline.ipynb` (section Jalon 4)
- Code : `src/evaluation.py` (métriques/plots)

## Jalon 5 — Architecture (choix MLP vs CNN vs RNN)

### Nature des données
Le problème principal est une **régression** sur des variables essentiellement **tabulaires** (pays, année, métriques d’usage/réseaux sociaux). Il n’y a pas de structure spatiale (pixels, cartes, grilles) ni de voisinage local naturel.

### Choix MLP (retenu)
Un **MLP** (réseau fully-connected) est un choix cohérent pour :
- des features tabulaires hétérogènes (numériques + encodages),
- capturer des non-linéarités avec une complexité contrôlable,
- fournir une baseline DL simple à entraîner et à comparer aux modèles ML.

### Pourquoi pas CNN (ici)
Les **CNN** exploitent des invariances locales (filtres, translation) et une **spatialité** (images, signaux 2D/3D). Dans ce dataset, les colonnes n’ont pas d’ordre spatial pertinent : appliquer des convolutions sur un vecteur de features serait arbitraire et rarement justifiable.

### RNN (possible) et choix “séquences”
Un **RNN** (LSTM/GRU) devient pertinent si on modélise explicitement des **séquences temporelles** (ex. historique par pays sur plusieurs années). Le projet explore cette dimension dans la partie “DL avancé” en construisant des séquences et en utilisant un modèle séquentiel (Transformer temporel) : même objectif qu’un RNN (dépendances temporelles), mais avec attention plutôt que récurrence.

### Où retrouver dans le projet
- MLP (baseline DL) : `src/models_dl.py` + `notebooks/03_dl_fondamental.ipynb`
- Séquences + modèle temporel : `src/models_advanced.py` + `notebooks/04_dl_avance.ipynb`

## Jalon 6 — Optimisation DL (hyperparamètres)

### Objectif
Améliorer la performance et la stabilité d’entraînement du MLP via une recherche d’hyperparamètres et des techniques anti-instabilité.

### Recherche d’hyperparamètres
- Méthode : Optuna (optimisation bayésienne / recherche guidée)
- Exemples d’hyperparamètres explorés : profondeur, largeurs des couches, `dropout`, `learning rate`, `weight_decay`

### Choix d’optimisation et stabilité gradient
- Optimiseur : Adam (bon compromis stabilité/vitesse)
- **BatchNorm** + **ReLU** : stabilisation des activations
- **Gradient clipping** : limite l’explosion des gradients
- Scheduler LR : réduction du LR sur plateau (ReduceLROnPlateau)

### Où retrouver dans le projet
- Notebook : `notebooks/03_dl_fondamental.ipynb` (section Jalon 6)
- Code : `src/models_dl.py` (training loop)

## Jalon 7 — Évaluation DL (comparaison et limites)

### Objectif
Comparer l’approche Deep Learning à la baseline ML, en gardant une évaluation cohérente (split identique, métriques comparables) et en discutant les limites.

### Résultats et comparaison
- Les modèles DL peuvent capturer des non-linéarités (au prix d’un risque d’overfitting plus élevé si dataset limité)
- Baselines ML (Ridge/Lasso) : plus simples, plus rapides, souvent très compétitives sur données tabulaires

### Limites et points d’attention
- Temps d’entraînement et variabilité (seed, initialisation)
- Sensibilité aux hyperparamètres
- Interprétabilité plus faible que les modèles linéaires

### Où retrouver dans le projet
- Notebooks : `notebooks/03_dl_fondamental.ipynb` (MLP) et `notebooks/04_dl_avance.ipynb` (Transformer)
- Code : `src/models_dl.py` et `src/models_advanced.py`
