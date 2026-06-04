# Transparence IA

Ce projet a été développé avec assistance IA (pair programming) pour accélérer la production, la structuration et la relecture.

## 1) Ce que l’IA a aidé à faire

- Structuration du projet (séparation `src/`, `tests/`, `app/`, `notebooks/`)
- Aide à l’implémentation (boucles d’entraînement, transformer temporel, métriques)
- Amélioration de la documentation (README, checklists)
- Mise en place CI (GitHub Actions)

## 2) Ce que l’IA ne remplace pas

- La compréhension : tout le code a été relu et expliqué
- La validation : exécution locale + tests + lancement Streamlit
- Les choix scientifiques : hypothèses et limites explicitement assumées

## 3) Limites / précautions

- Le dataset est synthétique : performances potentiellement trop optimistes
- L’extrapolation temporelle (après 2060) n’est pas une prédiction scientifique
- Les notebooks contiennent des sorties exécutées pour faciliter la correction, mais doivent rester reproductibles en rerun local

## 4) Traçabilité

- Dépôt Git avec historique de commits
- CI qui rejoue les tests sur push/PR
