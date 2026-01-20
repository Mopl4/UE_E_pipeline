# Pipeline (reproductible, H5-only)

Entry point :
- `uv run -m pipeline ...`

## Vue d’ensemble

Cette pipeline permet de :
- entraîner un stacking reproductible (base models → meta-learner) via OOF K-fold (sans leakage pour le meta-learner),
- activer/désactiver les base models : HGB (tabulaire), LSTM (temporel), CNN 1D (temporel), Chloe (multi-entrée EEG+meta),
- repartir **uniquement** du `X_*.h5` (`dataset_key=features`),
- gérer le déséquilibre de classes (undersampling équilibré ou poids auto),
- optimiser des hyperparamètres (random search sans dépendances) en choisissant quels modèles optimiser,
- sauvegarder un run complet (manifest + modèles + OOF + métriques) et faire `predict/evaluate/analyze` sans entraînement implicite.

Le pipeline est orienté “runs” :
- `train` crée un dossier `runs/<timestamp>/` avec `manifest.json`, modèles, prédictions OOF, métriques.
- `predict`, `evaluate`, `analyze` rechargent un run existant (pas d’entraînement implicite).

Par défaut :
- `train` sauvegarde un run (`--save` activé),
- et refit les modèles finaux (`--refit-final` activé) pour permettre `predict` ensuite.

## Flags (référence rapide)

| Commande | Flag | Description |
|---|---|---|
| `train` | `--run-dir` | Dossier de run (sinon `runs/<timestamp>`). |
| `train` | `--save/--no-save` | Sauvegarde ou non les artefacts du run. |
| `train` | `--refit-final/--no-refit-final` | Refit base models sur tout le train + sauvegarde (requis pour `predict`). |
| `train` | `--splits` | Nb folds OOF (plus grand = plus lent, plus stable). |
| `train` | `--timing` | Affiche les durées `fit/predict` par fold et par modèle. |
| `train` | `--x-h5` | H5 source (train). |
| `train` | `--y-csv` | Labels (train). |
| `train` | `--h5-dataset-key` | Clé du dataset H5 (par défaut `features`). |
| `train` | `--with-hgb/--no-with-hgb` | Active/désactive HGB (par défaut: ON). |
| `train` | `--with-lstm/--no-with-lstm` | Active/désactive LSTM (par défaut: OFF). |
| `train` | `--with-cnn/--no-with-cnn` | Active/désactive CNN (par défaut: OFF). |
| `train` | `--with-chloe/--no-with-chloe` | Active/désactive Chloe (par défaut: OFF). |
| `train` | `--hgb-fe` | HGB: feature engineering DWT (~110) + 11 meta. |
| `train` | `--hgb-meta-only` | HGB: garde uniquement les 11 meta features (drop signal brut). (Par défaut si HGB sans `--hgb-fe`.) |
| `train` | `--undersample-balanced` | Undersample pour obtenir autant d’exemples par classe (0/1/2). |
| `train` | `--class-weights-auto` | Poids auto (inverse fréquence), sans suppression. |
| `train` | `--lstm-tf-log-level` | `TF_CPP_MIN_LOG_LEVEL` (appliqué aux modèles TF: LSTM/CNN/Chloe). |
| `train/predict/evaluate/analyze` | `--cpu-only` | Force CPU-only (évite les tentatives d’init CUDA/`cuInit` de TensorFlow). |
| `train` | `--optimize` | Active la random search. |
| `train` | `--opt-targets` | Cibles: `meta hgb lstm cnn chloe`. |
| `train` | `--opt-budget-*-*` | Budget (epochs/samples) pendant trials (LSTM/CNN/Chloe). |
| `predict` | `--run-dir` | Run à utiliser. |
| `predict` | `--x-h5` | H5 à prédire (requis). |
| `predict` | `--out-format` | `full` (id,pred,proba_*) ou `benchmark` (id,label). |
| `evaluate` | `--run-dir` | Run à évaluer. |
| `analyze` | `--run-dir` | Run à analyser. |

## Valeurs par défaut (rappel)

### LSTM (si `--with-lstm`)
- `--lstm-epochs 3`, `--lstm-downsample 5`, `--lstm-max-train-samples 50000`
- `--lstm-units 64`, `--lstm-dense-units 32`
- `--lstm-batch-size 32`, `--lstm-predict-batch-size 1024`, `--lstm-verbose 1`

### CNN (si `--with-cnn`)
- `--cnn-epochs 20`, `--cnn-batch-size 64`, `--cnn-downsample 1`
- `--cnn-max-train-samples 0` (= pas de cap)
- `--cnn-predict-batch-size 4096`, `--cnn-verbose 1`

### Chloe (si `--with-chloe`)
- `--chloe-epochs 10`, `--chloe-batch-size 32`, `--chloe-predict-batch-size 2048`
- Archi par défaut = `lstm-cnn.py` (Conv1D 32/64, LSTM 64, meta dense 32, fusion dense 32).

## Toutes les combinaisons (base models)

Base models disponibles :
- HGB : `--with-hgb` (par défaut ON). Si `--hgb-fe` n’est pas activé, HGB utilise automatiquement le mode `meta-only` (drop du signal brut `11:1261`).
- LSTM : `--with-lstm`
- CNN : `--with-cnn`
- Chloe : `--with-chloe`

### 1) HGB seul (feature engineering)
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --no-with-cnn --no-with-chloe --hgb-fe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 3 --timing
```

### 2) HGB seul (meta-only)
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --no-with-cnn --no-with-chloe --hgb-meta-only \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 3 --timing
```

### 3) LSTM seul
```bash
uv run -m pipeline train --no-with-hgb --with-lstm --no-with-cnn --no-with-chloe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --timing
```

### 4) CNN seul
```bash
uv run -m pipeline train --no-with-hgb --no-with-lstm --with-cnn --no-with-chloe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --cnn-epochs 10 --cnn-max-train-samples 30000 --timing
```

### 5) Chloe seul
```bash
uv run -m pipeline train --no-with-hgb --no-with-lstm --no-with-cnn --with-chloe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --chloe-epochs 10 --timing
```

### 6) HGB + CNN
```bash
uv run -m pipeline train --with-hgb --hgb-fe --with-cnn --no-with-lstm --no-with-chloe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 \
  --cnn-epochs 20 --cnn-downsample 1 --cnn-batch-size 64 --timing
```

### 7) HGB + Chloe
```bash
uv run -m pipeline train --with-hgb --hgb-fe --with-chloe --no-with-lstm --no-with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --chloe-epochs 10 --timing
```

### 8) HGB + LSTM + CNN + Chloe
```bash
uv run -m pipeline train --with-hgb --hgb-fe --with-lstm --with-cnn --with-chloe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --cnn-epochs 3 --cnn-max-train-samples 10000 --chloe-epochs 3 --timing
```

## Predict (artefacts uniquement)

`predict` recharge uniquement les artefacts sauvegardés dans `runs/<RUN_ID>/` (pas d’entraînement implicite).

Si le run contient HGB + `--hgb-fe`, le feature engineering est recalculé à partir du `X_test.h5` (même traitement que pendant le train).

Si tu es sur CPU (pas de GPU), ajoute `--cpu-only` pour éviter les tentatives d’init CUDA de TensorFlow (message `failed call to cuInit ...`).

Exemple (sortie complète avec probabilités `proba_*`) :

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

- `evaluate` : calcule des métriques (accuracy, confusion matrix, etc.) sans modifier le run. Par défaut il s’appuie sur les prédictions OOF sauvegardées.
- `analyze` : post-mortem plus détaillé (normalisation de la confusion matrix + classification report) et peut écrire un résumé JSON dans `runs/<RUN_ID>/metrics/`.
