# Pipeline (reproductible, H5-only)

Entry point :
- `uv run -m pipeline ...`

## Vue d’ensemble

Cette pipeline permet de :
- entraîner un stacking reproductible (base models → meta-learner) via OOF K-fold (sans leakage pour le meta),
- activer/désactiver les base models : HGB (tabulaire), LSTM (temporel), CNN 1D (temporel),
- repartir **uniquement** du `X_*.h5` (`dataset_key=features`), avec deux modes HGB :
  - `--hgb-fe` : feature engineering DWT (~110 features) sur `features[:, 11:1261]`
  - `--hgb-meta-only` : drop du signal brut, garde seulement les 11 meta features
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
| `train` | `--with-hgb/--no-with-hgb` | Active/désactive HGB (par défaut: activé). |
| `train` | `--with-lstm/--no-with-lstm` | Active/désactive LSTM (par défaut: off). |
| `train` | `--with-cnn/--no-with-cnn` | Active/désactive CNN (par défaut: off). |
| `train` | `--hgb-fe` | HGB: feature engineering DWT (~110) + 11 meta. |
| `train` | `--hgb-fe-chunk-size` | Chunk pour featurization H5. |
| `train` | `--hgb-meta-only` | HGB: garde uniquement les 11 meta features (drop signal brut). |
| `train` | `--undersample-balanced` | Undersample pour obtenir autant d’exemples par classe. |
| `train` | `--undersample-seed` | Seed pour l’undersampling. |
| `train` | `--class-weights-auto` | Poids auto (inverse fréquence), sans suppression. |
| `train` | `--meta-C` | `C` du meta-learner (logreg). |
| `train` | `--optimize` | Active la random search. |
| `train` | `--opt-targets` | Cibles: `meta hgb lstm cnn`. |
| `train` | `--opt-trials`, `--opt-seed` | Nb essais + seed. |
| `train` | `--opt-budget-lstm-*`, `--opt-budget-cnn-*` | Budget (epochs/samples) pendant trials. |
| `train` | `--lstm-fast` | Preset vitesse LSTM (override). |
| `train` | `--cnn-load-only` | Charge un CNN `.keras` au lieu de ré-entraîner (attention leakage). |
| `predict` | `--run-dir` | Run à utiliser. |
| `predict` | `--x-h5` | H5 à prédire (requis). |
| `predict` | `--out` | CSV de sortie (`pred` + `proba_*`). |
| `evaluate` | `--run-dir` | Run à évaluer. |
| `evaluate` | (sans flags) | Évalue sur les OOF sauvegardés. |
| `evaluate` | `--x-h5 --y-csv` | Évalue sur un dataset fourni (via `predict`). |
| `analyze` | `--run-dir` | Run à analyser. |
| `analyze` | `--print-report/--no-print-report` | Affiche le classification report. |
| `analyze` | `--save-json/--no-save-json` | Sauve `metrics/analyze_oof.json`. |

## Valeurs par défaut (rappel)

### LSTM (si `--with-lstm`)
- `--lstm-epochs 3`, `--lstm-downsample 5`, `--lstm-max-train-samples 50000`
- `--lstm-units 64`, `--lstm-dense-units 32`
- `--lstm-batch-size 32`, `--lstm-predict-batch-size 1024`, `--lstm-verbose 1`
- `--lstm-fast` est désactivé ; si activé, il override les valeurs ci-dessus.

### CNN (si `--with-cnn`)
- `--cnn-epochs 20`, `--cnn-batch-size 64`, `--cnn-downsample 1`
- `--cnn-max-train-samples 0` (= pas de cap)
- `--cnn-predict-batch-size 4096`, `--cnn-verbose 1`
- `--cnn-load-only` est désactivé.

## Train (exemples)

### HGB (2 modes)

HGB + feature engineering (recommandé si tu veux exploiter le signal) :
```bash
uv run -m pipeline train --with-hgb --hgb-fe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 3 --timing
```
-> Reconstruit `(N, 121)` (11 meta + ~110 DWT) depuis le H5, puis OOF + meta + refit final.

HGB meta-only (très rapide, drop du signal brut) :
```bash
uv run -m pipeline train --with-hgb --hgb-meta-only \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 3 --timing
```
-> Entraîne HGB sur `(N, 11)` (meta features uniquement).

### CNN only (sans HGB)
```bash
uv run -m pipeline train --no-with-hgb --no-with-lstm --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --cnn-epochs 10 --cnn-max-train-samples 30000 --timing
```

### HGB + CNN
```bash
uv run -m pipeline train --with-hgb --hgb-fe --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --cnn-epochs 5 --cnn-max-train-samples 20000 --timing
```

## Toutes les combinaisons (base models)

Base models disponibles :
- HGB : `--with-hgb` (par défaut ON) + **obligatoire** : `--hgb-fe` ou `--hgb-meta-only`
- LSTM : `--with-lstm`
- CNN : `--with-cnn`

Pré-requis :
- Le pipeline est H5-only → toujours fournir `--x-h5` (et `--y-csv` en train).

### 1) HGB seul (feature engineering)
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --no-with-cnn --hgb-fe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 3 --timing
```

### 2) HGB seul (meta-only, drop du signal)
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --no-with-cnn --hgb-meta-only \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 3 --timing
```

### 3) LSTM seul
```bash
uv run -m pipeline train --no-with-hgb --with-lstm --no-with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --timing
```

### 4) CNN seul
```bash
uv run -m pipeline train --no-with-hgb --no-with-lstm --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --cnn-epochs 10 --cnn-max-train-samples 30000 --timing
```

### 5) HGB + LSTM
```bash
uv run -m pipeline train --with-hgb --with-lstm --no-with-cnn --hgb-fe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --timing
```

### 6) HGB + CNN
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --with-cnn --hgb-fe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --cnn-epochs 5 --cnn-max-train-samples 20000 --timing
```

### 7) LSTM + CNN (sans HGB)
```bash
uv run -m pipeline train --no-with-hgb --with-lstm --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --cnn-epochs 5 --cnn-max-train-samples 20000 --timing
```

### 8) HGB + LSTM + CNN
```bash
uv run -m pipeline train --with-hgb --with-lstm --with-cnn --hgb-fe \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --cnn-epochs 5 --cnn-max-train-samples 20000 --timing
```

Note : si tu veux HGB “léger” dans les combinaisons, remplace `--hgb-fe` par `--hgb-meta-only`.

## Déséquilibre de classes (2 options)

Deux stratégies possibles (mutuellement exclusives) :
- `--undersample-balanced` : supprime des exemples des classes surreprésentées (0/1) pour obtenir autant d’exemples par classe. Le run enregistre `balance/undersample_balanced.npz` avec `kept_idx`.
- `--class-weights-auto` : garde toutes les données mais pondère la loss (HGB via `sample_weight`, LSTM/CNN via `class_weight`).

Exemple undersample équilibré :
```bash
uv run -m pipeline train \
  --undersample-balanced --undersample-seed 42 \
  --with-hgb --hgb-fe --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --timing
```

Exemple poids auto :
```bash
uv run -m pipeline train \
  --class-weights-auto \
  --with-hgb --hgb-fe --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --timing
```

## Optimisation (random search)

```bash
uv run -m pipeline train \
  --with-hgb --hgb-fe --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 \
  --optimize --opt-targets meta hgb cnn \
  --opt-trials 20 --opt-seed 42 \
  --opt-budget-cnn-epochs 2 --opt-budget-cnn-max-train-samples 20000 \
  --timing
```
-> Écrit `opt/trials.jsonl`, `opt/best_params.json`, `opt/best_score.json`, `opt/config_used.json` dans le run.

## Predict (artefacts uniquement)

```bash
uv run -m pipeline predict --run-dir runs/<RUN_ID> --x-h5 data/X_test.h5 --out preds.csv
```

Important : un run fige la liste et l’ordre des base models (dans `manifest.json`). Au `predict`, tu dois réutiliser exactement ces modèles (même ordre). Si tu veux passer de 3 modèles à 2, il faut entraîner un nouveau run (OOF + meta) avec ces 2 modèles.

## Evaluate

Sur les OOF sauvegardés du run :
```bash
uv run -m pipeline evaluate --run-dir runs/<RUN_ID>
```

Sur un dataset (in-sample si tu évalues sur train) :
```bash
uv run -m pipeline evaluate --run-dir runs/<RUN_ID> --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv
```

## Analyze (post-mortem)

```bash
uv run -m pipeline analyze --run-dir runs/<RUN_ID>
```
-> Affiche un résumé du run (base_model_order, best_params/meta, imbalance), puis confusion matrix (brute + normalisée) et métriques (accuracy/balanced/F1 + report).
