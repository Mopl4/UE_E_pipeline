# Pipeline (reproductible, H5-only)

Entry point :
- `uv run -m pipeline ...`

## Vue d’ensemble

- `train` entraîne un stacking (base models → meta-learner) via OOF K-fold, puis (par défaut) refit et sauvegarde les modèles finaux dans `runs/<RUN_ID>/models/`.
- `predict`, `evaluate`, `analyze` rechargent un run existant (pas d’entraînement implicite).
- Le pipeline repart du H5 (`features`) : HGB doit utiliser `--hgb-fe` (DWT) ou `--hgb-meta-only` (11 meta, signal droppé).

## Predict (artefacts uniquement)

Exemple (sortie complète avec probabilités) :

```bash
uv run -m pipeline predict \
  --run-dir runs/<RUN_ID> \
  --x-h5 data/X_test.h5 \
  --out preds.csv
```

Important : un run fige la liste et l’ordre des base models (dans `manifest.json`). Au `predict`, tu dois réutiliser exactement ces modèles (même ordre). Si tu veux passer de 3 modèles à 2, il faut entraîner un nouveau run.

## Soumission (benchmark)

Pour soumettre, on réutilise les poids/artefacts sauvegardés lors d’un entraînement (`runs/<RUN_ID>/models/*`) et on génère un CSV au format demandé `id,label` (sans ré-entraîner) :

```bash
uv run -m pipeline predict \
    --run-dir runs/<RUN_ID> \
    --x-h5 data/X_test.h5 \
    --out y_benchmark.csv \
    --out-format benchmark
```

## Evaluate / Analyze

- `evaluate` : score rapide (OOF par défaut), ou évalue sur un dataset fourni.
- `analyze` : post-mortem plus détaillé (confusion matrix, métriques, report), et peut écrire un JSON dans `runs/<RUN_ID>/metrics/`.

