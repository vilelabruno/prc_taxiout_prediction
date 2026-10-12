# PRC 2026 taxi-out prediction

Two-level residual model for predicting the taxi-out time (`TAXITIME_SEC_mvt`) of departures at ten European airports
(EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM). The final run scores the January, February, June and July 2026
departures of `final_ranking.parquet`.

This repository contains the code to train, cross-validate and predict, plus the trained artifacts that are small enough to
version (models, out-of-fold predictions, per-component predictions). Large tabular inputs (`*.parquet`) are not versioned.

## Approach

1. **Level 1, two base models** trained on all of 2025:
   - Base A: a chain of CatBoost and LightGBM models fitted on the residual against a schedule-based baseline
     (`prc/basea`).
   - Base B: an ensemble of LightGBM, CatBoost and an MLP defined by a YAML config, followed by a refit of the segment of
     LIRF departures without a matched flight plan (`prc/baseb`).
   - `v36 = 0.5 * A + 0.5 * B` (`prc/blend.py`).
2. **Level 2, residual layers** applied on top of `v36` (`prc/layers`, `prc/otimista`):
   rule and model layers for LIRF, LTFM and LEBL, and an ADS-B based corrector ("layer D") for EDDF, EDDM, EHAM, LSZH,
   LEBL, EGLL and LEMD, trained on neighbouring months of 2025 with a seasonal-distance weighting.
3. **Stack** (`prc/stack`): ridge/lasso combiners fitted per airport cell on out-of-fold 2025 predictions, a small
   population of neural combiners, and 15 members (base-A and layer deltas, five LightGBM variants, five sequence networks, an MLP).
   Safety rules fall back to the layer output where the stack correction is out of scale.
4. **Residual specialists** (`prc/espera_rot`): a LightGBM on aircraft rotation, stand-slot and historical stand profiles
   for non-LIRF airports, and a separate LIRF-only specialist.
5. **Post-processing rules** (`prc/layers/lirf_night_rule.py`, `prc/layers/lirf_eco_rule.py`): fixed rules for LIRF departures
   without an actual off-block time, applied in a 50/50 blend with the model output.

## Repository layout

```
prc/paths.py            all input/output locations (environment variables below)
prc/basea               base A: features, component fits, assembly, out-of-fold validation
prc/baseb               base B: trimmed model framework, YAML configs, build/holdout/refit CLI
prc/adsb                ADS-B surface extraction and per-flight feature extraction
prc/layers              level-2 layers, layer composition, LIRF rules
prc/otimista            layer D: contexts, leave-month-out validation, application, H_base members
prc/stack               member table, combiners, neural population, final stack application
prc/espera_rot          residual specialists (all airports, and LIRF only)
prc/blend.py            v36 blend of the two bases
prc/compose.py          assembly of the final pre-specialist prediction for the four months
prc/terms.py            builds delta terms between two versions of a component
prc/pipeline.py         stage definitions used by the three entry points
scripts/train.py        runs training stages
scripts/cv.py           runs validation stages
scripts/predict.py      runs prediction/composition stages
artifacts/              versioned models and out-of-fold predictions (see docs/ARTIFACTS.md)
docs/                   stage reference and artifact manifest
```

## Setup

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.11 or newer. CatBoost and LightGBM use CPU; the sequence networks and the MLP run on CPU, CUDA or Apple MPS
(`--device`).

## Data and locations

`prc/paths.py` reads these environment variables (defaults relative to the repository):

| Variable | Default | Content |
|---|---|---|
| `PRC_DATA_ROOT` | `data/` | `raw/` monthly training files, `final/final_ranking.parquet`, `final/final_submitting.parquet`, `processed/` |
| `PRC_RANKING`, `PRC_SUBMITTING` | `data/final/...` | ranking and submission template |
| `PRC_TRAINING_DIR` | `data/raw` | the twelve `training_2025-*.parquet` files |
| `PRC_SURFACE_DIR` | `data/processed/adsb_surface` | daily ADS-B surface extracts |
| `PRC_WORK` | `work/` | intermediates |
| `PRC_ARTIFACTS` | `artifacts/` | models, out-of-fold predictions, references |

Labels are used only from 2025. The ranking files are read for features and identifiers; no label of the scored months is
read.

## Running

All entry points accept `--dry-run`, which prints the commands without executing them.

```
python scripts/train.py features basea baseb blend layers otimista v53 stack espera
python scripts/cv.py basea baseb layers otimista stack espera
python scripts/predict.py compose espera
```

`python scripts/<entry>.py all` runs every stage of that entry point. Stage definitions and the exact commands are in
`docs/STAGES.md`. The final file is `artifacts/submission_final.parquet`, with columns `MVT_ID_mvt` and `TAXITIME_SEC_mvt`.

## Months and composition

February and June 2026 have no 2025 hold-out of their own, so their layer-D models use the other eleven months of 2025
with seasonal-distance weights and a proxy prediction for the own month. January and July use the 2025 hold-out of the
corresponding month.

For January and July the prediction is assembled as a reference layer plus model deltas: the reference is a prediction file
for these rows from an earlier stage of the work (`artifacts/reference/`, not versioned), and the deltas are the differences
between the retrained models and the earlier models for the same rows (stack, ADS-B layer D, pool, LIRF physical layer, and
the receita-D term, all delivered as precomputed delta files; `prc/terms.py` builds them). The corrections that remove
row-level values from that reference are also delivered as files (`d_row_adjustments_removed`,
`set_model_values_on_adjusted_rows`). February and June use the retrained model chain only.

## Validation evidence

Each stage was compared against the output it replaces, using the original inputs:

| Stage | Result |
|---|---|
| Base A assembly (`prc/basea/assemble.py`) | matches a literal transcription of the original mixing rules to 1.5e-11 s |
| Non-ADS-B layers | identical to the reference implementation (0.0 s over 670,790 rows) |
| Layer D (S4 and pooled models) | identical to the reference implementation (0.0 s, same eligibility mask) |
| Stack application | reproduces the earlier stack output exactly (0.0 s over 344,841 rows) |
| Residual specialist (all airports) | matches within 0.77 s maximum, 0.0076 s RMS |
| LIRF specialist | matches to 1e-12 s; cross-validated gain (jan/jul) of -8.54 s RMSE at dose 0.5 |
| Out-of-fold combiner | nested OOF RMSE outside LIRF: 186.39 s against 190.01 s for the hand-built stack |

Leave-month-out validation of layer D and the combiner uses 2025 labels only.

## Reproducibility status and limitations

- **Base A and base B weights.** The versioned artifacts hold the per-component predictions of the full fit
  (`artifacts/base_a_full_final`), not the fitted models. The current `fit` entry points persist models, so a rerun
  stores them under `artifacts/models/`.
- **Base A.** The base A used here is the chain implemented in `prc/basea`. An earlier base-A implementation maintained by a
  collaborator is not part of this repository, and the first half of the level-1 blend is this chain.
- **Not versioned inputs.** `*.parquet` files (raw data, ADS-B extracts, hold-outs, reference files and delta files) are
  not in the repository. `docs/STAGES.md` lists which stage reads which file.
- **Terms delivered as files.** The generating scripts for the receita-D, pooled-Huber and LIRF physical-layer delta terms
  of January/July are not part of this repository.
- **Selection of constants.** Several constants (the blend weights of the reference layer, some doses and thresholds, and the
  choice between specialist variants) were selected with internal validation and also with feedback from the public
  leaderboard. They are listed in `docs/STAGES.md`. February and June contain no layer fitted on leaderboard feedback.
- **February/June extrapolation.** The stack combiners and specialists were fitted on January and July 2025 out-of-fold
  predictions; their behaviour on February and June is extrapolation, and cannot be measured without labels.
- **Rules.** The two LIRF post-processing rules rest on small samples in 2025 (35 and 43 flights).
