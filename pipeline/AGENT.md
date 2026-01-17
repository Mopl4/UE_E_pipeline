# pipeline/AGENT.md

## Objectif

Refactoriser `stacking/stacking.py` en une pipeline ML propre, reproductible et orientée "runs":
- entraînement (OOF stacking + refit final),
- évaluation sans modifier le run,
- prédiction en chargeant uniquement des artefacts existants.

## Règles (invariants)
- Splits par indices uniquement (pas de réordonnancement silencieux).
- Meta-learner entraîné sur OOF uniquement (pas de leakage).
- En inférence, pas d’entraînement implicite (on charge des artefacts).
- Ordre des base models figé et sauvegardé (concat `Z` stable).
- LSTM optionnel, activable/désactivable via CLI.

## Arborescence
- `pipeline/cli.py` : commandes `train|evaluate|predict` (entrypoint).
- `pipeline/data.py` : chargement CSV/H5 + validations de shape.
- `pipeline/stacking_oof.py` : logique OOF par indices (retourne blocs, `Z_oof`, `fold_id`).
- `pipeline/artifacts.py` : sauvegarde/chargement (manifest + modèles).
- `pipeline/metrics.py` : métriques (accuracy, confusion matrix, rapports).
- `pipeline/models/` : wrappers modèles base + meta (HGB, LSTM, meta).
- `pipeline/feature_engineering/` : génération des features boosting (H5 -> tabulaire).

## Conventions
- Les runs sont écrits dans `runs/<timestamp>` (configurable via `--run-dir`).
- Les artefacts modèles sont écrits uniquement pour le “best” en cas d’optimisation.
- Feature engineering HGB:
  - entrée: `X_*.h5` dataset `features` (shape `(N, 1261)`)
  - on garde les 11 premières colonnes (meta statiques)
  - on remplace le signal brut `features[:, 11:1261]` (1250 points) par ~110 features :
    - Haar DWT stats (logique `dwt_features.py`)
    - DWT `db4` + stats par fenêtres (logique `Ethan_P/feature_engineering/dwt_max.py`)
  - sortie HGB: tableau `(N, 121)` = `11 + 110` (pas de points bruts du signal)

## Optimisation
- `--optimize` (random search) enregistre `opt/trials.jsonl`, `opt/best_params.json`, `opt/best_score.json`, `opt/config_used.json`.
- `--opt-targets` permet de choisir quoi optimiser (`meta`, `hgb`, `lstm`, `cnn`).
- Budget optimisation (pour éviter des runs trop longs) :
  - LSTM: `--opt-budget-lstm-epochs`, `--opt-budget-lstm-max-train-samples`
  - CNN: `--opt-budget-cnn-epochs`, `--opt-budget-cnn-max-train-samples`
- Invariant: le meta-learner reste entraîné uniquement sur `Z_oof` (OOF), même pendant l’optimisation.

## Déséquilibre de classes
- `--undersample-balanced` : construit un sous-ensemble équilibré (mêmes effectifs par classe) *avant* OOF. Le run sauvegarde `balance/undersample_balanced.npz` avec `kept_idx` (reproductible).
- `--class-weights-auto` : conserve toutes les données, mais utilise des poids de classe (inverse fréquence) pendant l'entraînement (HGB via `sample_weight`, LSTM/CNN via `class_weight`).
- Les deux options sont mutuellement exclusives.
