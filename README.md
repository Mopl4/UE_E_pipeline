# Pipeline (reproductible)

Entry point:
- `python -m pipeline ...`

## Vue d’ensemble

Cette pipeline permet de :
- entraîner un modèle de stacking reproductible (base models → meta-learner) avec OOF K-fold (sans data leakage au niveau du meta-learner),
- combiner plusieurs base models au choix : HGB (tabulaire), LSTM (temporel), CNN 1D (temporel),
- générer des features HGB depuis le H5 (feature engineering DWT) ou utiliser un CSV tabulaire existant,
- gérer le déséquilibre de classes (undersampling équilibré ou poids de classe automatiques),
- optimiser des hyperparamètres (random search sans dépendances) en choisissant quels modèles optimiser,
- sauvegarder un run complet (manifest + modèles + OOF + métriques) et faire `predict/evaluate` sans entraînement implicite.

Le pipeline est orienté “runs” :
- `train` crée un dossier `runs/<timestamp>/` avec `manifest.json`, modèles, prédictions OOF, métriques.
- `predict` et `evaluate` rechargent un run existant (pas d’entraînement implicite).
- `analyze` produit une analyse lisible (confusion matrix + métriques) à partir des prédictions sauvegardées.

Par défaut :
- `train` sauvegarde un run (`--save` activé),
- et refit les modèles finaux (`--refit-final` activé) pour permettre `predict` ensuite.

## Flags (référence rapide)

Tableau synthétique des flags les plus importants. Pour le détail complet, voir `uv run -m pipeline train -h` etc.

| Commande | Flag | Description |
|---|---|---|
| `train` | `--run-dir` | Dossier de run (sinon `runs/<timestamp>`). |
| `train` | `--save/--no-save` | Sauvegarde ou non les artefacts du run. |
| `train` | `--refit-final/--no-refit-final` | Refit chaque base model sur tout le train puis sauvegarde (nécessaire pour `predict`). |
| `train` | `--splits` | Nombre de folds OOF (plus grand = plus lent, plus stable). |
| `train` | `--timing` | Affiche les durées `fit/predict` par fold et par modèle. |
| `train` | `--with-hgb/--no-with-hgb` | Active/désactive HGB (par défaut activé). |
| `train` | `--with-lstm/--no-with-lstm` | Active/désactive LSTM (H5 requis si activé). |
| `train` | `--with-cnn/--no-with-cnn` | Active/désactive CNN (H5 requis si activé). |
| `train` | `--x-csv` | CSV tabulaire (utilisé par HGB si `--hgb-fe` n’est pas activé). |
| `train` | `--y-csv` | Labels (obligatoire en train). |
| `train` | `--x-h5` | H5 source (requis pour LSTM/CNN et pour `--hgb-fe`). |
| `train` | `--h5-dataset-key` | Clé du dataset H5 (par défaut `features`). |
| `train` | `--drop-cols` | Colonnes ignorées dans le CSV (ex: `id`, `label`). |
| `train` | `--hgb-fe` | Reconstruit les features HGB depuis le H5 (11 meta + ~110 DWT combo). |
| `train` | `--hgb-fe-chunk-size` | Taille de batch pour featurization H5. |
| `train` | `--undersample-balanced` | Undersample pour obtenir autant d’exemples par classe (0/1/2), avant tout entraînement. |
| `train` | `--undersample-seed` | Seed pour l’undersampling. |
| `train` | `--class-weights-auto` | Poids de classe auto (inverse fréquence), sans supprimer de données. |
| `train` | `--meta-C` | Paramètre `C` du meta-learner (logreg). |
| `train` | `--optimize` | Active la random search. |
| `train` | `--opt-targets` | Cibles à optimiser: `meta hgb lstm cnn`. |
| `train` | `--opt-trials` | Nombre d’essais. |
| `train` | `--opt-seed` | Seed de l’optimisation. |
| `train` | `--opt-budget-lstm-epochs` | Budget epochs LSTM pendant trials. |
| `train` | `--opt-budget-lstm-max-train-samples` | Budget nb samples LSTM pendant trials. |
| `train` | `--opt-budget-cnn-epochs` | Budget epochs CNN pendant trials. |
| `train` | `--opt-budget-cnn-max-train-samples` | Budget nb samples CNN pendant trials. |
| `train` | `--lstm-fast` | Preset vitesse LSTM (override). |
| `train` | `--lstm-downsample` | Sous-échantillonnage LSTM (plus grand = plus rapide). |
| `train` | `--lstm-epochs` | Epochs LSTM (hors optimisation). |
| `train` | `--lstm-max-train-samples` | Cap samples LSTM par fold (0 = pas de cap). |
| `train` | `--lstm-units` | Unités LSTM. |
| `train` | `--lstm-dense-units` | Unités Dense après LSTM. |
| `train` | `--lstm-batch-size` | Batch size LSTM. |
| `train` | `--lstm-predict-batch-size` | Batch size `predict_proba` LSTM. |
| `train` | `--lstm-verbose` | Verbose Keras LSTM. |
| `train` | `--lstm-load-model` | Charge un modèle LSTM `.keras` (pas d’entraînement du LSTM). |
| `train` | `--cnn-epochs` | Epochs CNN (si entraîné). |
| `train` | `--cnn-batch-size` | Batch size CNN. |
| `train` | `--cnn-downsample` | Sous-échantillonnage CNN. |
| `train` | `--cnn-max-train-samples` | Cap samples CNN par fold (0 = pas de cap). |
| `train` | `--cnn-predict-batch-size` | Batch size `predict_proba` CNN. |
| `train` | `--cnn-verbose` | Verbose Keras CNN. |
| `train` | `--cnn-load-only` | Charge `--cnn-load-model` et ne ré-entraîne pas le CNN (attention leakage). |
| `train` | `--cnn-load-model` | Chemin du modèle CNN `.keras`. |
| `predict` | `--run-dir` | Run à utiliser (obligatoire). |
| `predict` | `--x-csv` | CSV à prédire (requis si HGB=CSV dans le run). |
| `predict` | `--x-h5` | H5 à prédire (requis si LSTM/CNN et/ou HGB-fe). |
| `predict` | `--out` | Fichier CSV de sortie (pred + proba). |
| `evaluate` | `--run-dir` | Run à évaluer (obligatoire). |
| `evaluate` | `--x-csv/--y-csv` | Évalue sur un dataset fourni (sinon utilise les OOF sauvegardés). |
| `evaluate` | `--x-h5` | H5 si nécessaire (run avec LSTM/CNN ou HGB-fe). |
| `analyze` | `--run-dir` | Run à analyser (obligatoire). |
| `analyze` | `--source` | Source à analyser (`oof` pour l’instant). |
| `analyze` | `--print-report/--no-print-report` | Affiche le classification report. |
| `analyze` | `--save-json/--no-save-json` | Écrit `metrics/analyze_oof.json` dans le run. |

## Train (OOF stacking)

## Valeurs par défaut (rappel)

Ces valeurs s’appliquent si tu ne passes pas de flags spécifiques.

### LSTM (si `--with-lstm`)
- `--lstm-epochs 3`, `--lstm-downsample 5`, `--lstm-max-train-samples 50000`
- `--lstm-units 64`, `--lstm-dense-units 32`
- `--lstm-batch-size 32`, `--lstm-predict-batch-size 1024`, `--lstm-verbose 1`
- `--lstm-fast` est désactivé ; si activé, il override les valeurs ci-dessus.

### CNN (si `--with-cnn`)
- `--cnn-epochs 20`, `--cnn-batch-size 64`, `--cnn-downsample 1`
- `--cnn-max-train-samples 0` (= pas de cap, utilise tout le fold)
- `--cnn-predict-batch-size 4096`, `--cnn-verbose 1`
- `--cnn-load-only` est désactivé ; si activé, le CNN est chargé depuis `--cnn-load-model` et n’est pas ré-entraîné.

### HGB (2 modes)

HGB depuis un CSV tabulaire existant (rapide, pas de H5) :
```bash
uv run -m pipeline train --x-csv data/X_train_merged.csv --y-csv data/y_train_2.csv --splits 3
```
-> Entraîne HGB en OOF (3 folds), entraîne le meta-learner sur les probas OOF, puis refit les modèles finaux (par défaut).

HGB depuis le H5 + feature engineering (reconstruit les features à partir du signal EEG, sans garder les 1250 points bruts) :
```bash
uv run -m pipeline train --hgb-fe --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 3
```
-> Calcule un tableau tabulaire `(N, 121)` depuis le H5 (11 meta + ~110 DWT), puis même logique OOF/meta/refit.

Dans ce mode `--hgb-fe`, on prend `features[:, 11:1261]` (1250 points) et on le remplace par ~110 features (Haar DWT + db4+fenêtres), puis HGB s'entraîne sur `(N, 121)` = `11 meta` + `110 engineered`.

### HGB + LSTM (modèle temporel, optionnel)

Rapide (debug / itération) :
```bash
uv run -m pipeline train --with-lstm --x-h5 data/X_train.h5 --splits 2 --lstm-fast --timing
```
-> Ajoute un base model LSTM (sur le H5) au stacking; `--lstm-fast` réduit fortement le coût (downsample + cap + peu d’epochs).

Plus “sérieux” (plus lent) :
```bash
uv run -m pipeline train \
  --with-lstm \
  --x-h5 data/X_train.h5 \
  --splits 3 \
  --lstm-downsample 5 \
  --lstm-epochs 3 \
  --lstm-max-train-samples 50000 \
  --lstm-units 64 \
  --lstm-dense-units 32 \
  --lstm-batch-size 64 \
  --lstm-predict-batch-size 2048 \
  --timing
```
-> Même chose mais avec plus de capacité/epochs (souvent meilleur, plus long).

Notes perf LSTM :
- `--lstm-downsample` et `--lstm-max-train-samples` sont les deux leviers les plus efficaces.
- `--lstm-fast` override les paramètres manuels (preset vitesse).

### HGB + CNN 1D (modèle temporel, optionnel)

Sans data leakage (retrain par fold, plus lent) :
```bash
uv run -m pipeline train \
  --with-cnn \
  --x-h5 data/X_train.h5 \
  --splits 3 \
  --cnn-epochs 5 \
  --cnn-batch-size 64 \
  --cnn-max-train-samples 20000 \
  --cnn-predict-batch-size 4096 \
  --cnn-verbose 1 \
  --timing
```
-> Le CNN est entraîné séparément dans chaque fold OOF (pas de fuite) puis stacké via le meta-learner.

Load-only (très rapide, mais attention au leakage) :
```bash
uv run -m pipeline train \
  --with-cnn \
  --x-h5 data/X_train.h5 \
  --cnn-load-only \
  --cnn-load-model stacking/cnn_sleep_model.keras \
  --splits 2 \
  --timing
```

⚠️ `--cnn-load-only` est correct uniquement si le modèle chargé a été entraîné **sans voir** les exemples utilisés en validation OOF (sinon il peut y avoir fuite).

### Combinaisons

HGB + LSTM + CNN :
```bash
uv run -m pipeline train \
  --hgb-fe \
  --with-lstm --lstm-fast \
  --with-cnn --cnn-epochs 3 --cnn-max-train-samples 10000 \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --timing
```
-> Stacking à 3 base models; `Z_oof` = concat des probas OOF dans l’ordre sauvegardé dans `manifest.json`.

### Stacking sans HGB (optionnel)

CNN only (toujours avec meta-learner, mais `Z_oof` ne contient que les probas CNN) :
```bash
uv run -m pipeline train \
  --no-with-hgb \
  --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --cnn-epochs 10 --cnn-batch-size 64 --cnn-downsample 1 --cnn-max-train-samples 30000 \
  --timing
```

## Optimisation (simple)

`--optimize` lance une random search (sans dépendances) et enregistre les essais dans le run :
- `opt/trials.jsonl` : tous les essais (params + score)
- `opt/best_params.json` : meilleurs params trouvés
- `opt/best_score.json` : meilleur score (accuracy OOF)
- `opt/config_used.json` : config de l’optimisation (targets, budgets, seed, etc.)

Tu choisis quoi optimiser via `--opt-targets` parmi : `meta`, `hgb`, `lstm`, `cnn`.
Pour garder l’optimisation tractable, LSTM/CNN utilisent un budget réduit pendant les trials (`--opt-budget-*`).

```bash
uv run -m pipeline train \
  --x-csv data/X_train_merged.csv --y-csv data/y_train_2.csv \
  --splits 3 \
  --optimize --opt-targets meta \
  --opt-trials 30 --opt-seed 42
```

Optimiser plusieurs modèles (ex: `meta + hgb + lstm`) :
```bash
uv run -m pipeline train \
  --hgb-fe --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --with-lstm \
  --splits 2 \
  --optimize --opt-targets meta hgb lstm \
  --opt-trials 20 --opt-seed 42 \
  --opt-budget-lstm-epochs 1 --opt-budget-lstm-max-train-samples 20000 \
  --timing
```

## Predict (artefacts uniquement)

```bash
uv run -m pipeline predict --run-dir runs/<RUN_ID> --x-csv data/X_test_merged.csv --x-h5 data/X_test.h5 --out preds.csv
```
-> Recharge les modèles finaux sauvegardés dans `runs/<RUN_ID>/models/` et produit `preds.csv`.

But :
- faire de l’inférence “pure” (aucun entraînement),
- garantir la reproductibilité (mêmes artefacts → mêmes prédictions),
- produire un fichier exploitable (classe + probabilités).

Entrées à fournir (en fonction du run) :
- Si le run contient `lstm_temporal` ou `cnn_temporal` → `--x-h5` est obligatoire.
- Si le run contient `hgb_tabular` entraîné depuis CSV → `--x-csv` est obligatoire.
- Si le run contient `hgb_tabular` avec `--hgb-fe` → `--x-h5` suffit (features HGB reconstruites depuis le H5).

Sortie :
- `--out preds.csv` contient `id` (= index), `pred` (= argmax), et `proba_0/1/2`.

Si le run a été entraîné avec `--hgb-fe`, tu peux prédire sans CSV :
```bash
uv run -m pipeline predict --run-dir runs/<RUN_ID> --x-h5 data/X_test.h5 --out preds.csv
```
-> Ici `--x-h5` est suffisant car les features HGB sont reconstruites depuis le H5.

Notes :
- Si le run inclut `--with-lstm` ou `--with-cnn`, il faut fournir `--x-h5`.
- `predict` nécessite des modèles finaux (base + meta) sauvegardés : garder `--refit-final` activé au `train`.
- Important : un run fige la liste et l’ordre des base models (dans `manifest.json`). Au `predict`, tu dois réutiliser exactement ces modèles (même ordre), sinon la dimension de `Z` ne correspond plus au meta-learner et la prédiction n’a pas de sens. Si tu veux passer de 3 modèles à 2, il faut entraîner un nouveau run (OOF + meta) avec ces 2 modèles.

## Evaluate

Sur les OOF sauvegardés du run :
```bash
uv run -m pipeline evaluate --run-dir runs/<RUN_ID>
```
-> Recalcule des métriques depuis les prédictions OOF sauvegardées (rapide, sans entraînement, ne modifie pas le run).

But :
- mesurer la performance “out-of-fold” (pas de leakage),
- obtenir une matrice de confusion et des erreurs par classe,
- comparer plusieurs runs sur une base cohérente.

Sur un dataset (nécessite les modèles finaux sauvegardés via `--refit-final`) :
```bash
uv run -m pipeline evaluate --run-dir runs/<RUN_ID> --x-csv data/X_train_merged.csv --y-csv data/y_train_2.csv --x-h5 data/X_train.h5
```
-> Lance une prédiction avec les artefacts du run sur le dataset fourni, puis compare à `y_csv`.

Note :
- Évaluer sur `X_train` complet avec les modèles refit est “in-sample” (optimiste). Préfère l’OOF pour juger la généralisation.

## Analyze (post-mortem d’un run)

`analyze` sert à produire une analyse plus complète après un entraînement, sans ré-entraîner et sans modifier les modèles.

Exemple (analyse des OOF) :
```bash
uv run -m pipeline analyze --run-dir runs/<RUN_ID>
```

Sorties typiques :
- résumé du run (ordre des base models, meta/best_params, option imbalance si présente)
- accuracy, balanced accuracy, F1 macro/weighted
- confusion matrix + version normalisée (par classe vraie)
- taux d’erreur par classe
- classification report (precision/recall/f1) si activé

Fichiers :
- écrit `runs/<RUN_ID>/metrics/analyze_oof.json` (désactivable via `--no-save-json`)

## Toutes les combinaisons (base models)

Les base models possibles sont : HGB (`--with-hgb`, activé par défaut), LSTM (`--with-lstm`), CNN (`--with-cnn`).
Ci-dessous, des commandes “templates” pour entraîner un run avec chaque combinaison (le meta-learner est toujours entraîné sur OOF).

Pré-requis :
- Combinaisons impliquant LSTM/CNN : fournir `--x-h5 data/X_train.h5`
- HGB sans `--hgb-fe` : fournir `--x-csv data/X_train_merged.csv`
- HGB avec `--hgb-fe` : fournir `--x-h5 data/X_train.h5` (et **pas** besoin de `--x-csv`)

### 1) HGB seul (CSV)
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --no-with-cnn \
  --x-csv data/X_train_merged.csv --y-csv data/y_train_2.csv --splits 3 --timing
```

### 2) HGB seul (H5 + feature engineering)
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --no-with-cnn \
  --hgb-fe --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 3 --timing
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
uv run -m pipeline train --with-hgb --with-lstm --no-with-cnn \
  --hgb-fe --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --timing
```

### 6) HGB + CNN
```bash
uv run -m pipeline train --with-hgb --no-with-lstm --with-cnn \
  --hgb-fe --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --cnn-epochs 10 --cnn-max-train-samples 30000 --timing
```

### 7) LSTM + CNN (sans HGB)
```bash
uv run -m pipeline train --no-with-hgb --with-lstm --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --cnn-epochs 5 --cnn-max-train-samples 20000 --timing
```

### 8) HGB + LSTM + CNN
```bash
uv run -m pipeline train --with-hgb --with-lstm --with-cnn \
  --hgb-fe --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv --splits 2 --lstm-fast --cnn-epochs 5 --cnn-max-train-samples 20000 --timing
```

## Durées / profils recommandés

- Très rapide (sanity check) :
  - `--splits 2` + `--with-lstm --lstm-fast` et/ou `--with-cnn --cnn-max-train-samples 5000 --cnn-epochs 1`
- Intermédiaire :
  - `--splits 3`, LSTM `--lstm-downsample 10 --lstm-epochs 1 --lstm-max-train-samples 20000`
- Plus “sérieux” :
  - `--splits 5`, LSTM/CNN sans cap trop agressif (peut être long).

## Déséquilibre de classes (2 options)

Deux stratégies possibles (mutuellement exclusives) :
- `--undersample-balanced` : supprime des exemples des classes surreprésentées pour obtenir autant d’exemples par classe (0/1/2). Le run enregistre `balance/undersample_balanced.npz` avec `kept_idx` pour reproductibilité.
- `--class-weights-auto` : garde toutes les données, mais pondère la loss (poids inverses aux fréquences de classes).

Notes :
- Si tu actives `--undersample-balanced` et que HGB est activé, il faut entraîner HGB depuis le H5 (`--hgb-fe`) pour que tous les modèles repartent du même sous-ensemble d’indices.

Exemples :

Undersampling équilibré (réduit surtout la classe 0 pour matcher la classe minoritaire) :
```bash
uv run -m pipeline train \
  --undersample-balanced --undersample-seed 42 \
  --hgb-fe --with-hgb --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --timing
```

Poids auto (garde toutes les données, mais pondère la loss) :
```bash
uv run -m pipeline train \
  --class-weights-auto \
  --hgb-fe --with-hgb --with-cnn \
  --x-h5 data/X_train.h5 --y-csv data/y_train_2.csv \
  --splits 2 --timing
```
