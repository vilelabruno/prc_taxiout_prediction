# Stage reference

Commands below are printed by `python scripts/<entry>.py <stage> --dry-run`; paths are relative to the repository.
Every stage reads and writes under the locations in `prc/paths.py`.

## Training stages (`scripts/train.py`)

| Stage | What it produces | Main inputs |
|---|---|---|
| `features` | ADS-B per-flight, position, stand, off-block, leg and route features for 2025 and for the four 2026 months | daily surface extracts, `final_ranking.parquet`, `training_all_2025.parquet` |
| `basea` | fits the base-A components, stores their predictions (and, on a rerun, their models), assembles base A | `training_*.parquet`, `final_ranking.parquet` |
| `baseb` | refits base B on all of 2025 and applies the segment refit | `training_*.parquet`, `final_ranking.parquet`, `prc/baseb/configs/*.yaml` |
| `blend` | `v36_final.parquet` = 0.5 A + 0.5 B | outputs of `basea` and `baseb` |
| `layers` | non-ADS-B layers (LIRF rule, flight copy, LTFM delay state, LIRF unmatched mixture) and the LEBL chain | `v36_final.parquet`, 2025 labels, hold-out, METAR events |
| `otimista` | layer D for February and June, H_base members | `v36_final.parquet`, ADS-B features, 2025 hold-out and neighbour months |
| `v53` | composes the layer outputs into the pre-stack prediction | outputs of `layers`, `otimista` |
| `stack` | applies the reduced stack combiner | member table, `perm_coefs_nonets.pkl` |
| `stack_full` | builds the member table, the neural networks and the MLP, fits the combiners, applies the full stack | all member inputs |
| `espera` | specialists and LIRF rules, final file | `pre_espera_final.parquet`, 2025 training data |

## Entry points

### `scripts/train.py all`

```
python prc/adsb/own_flight_features.py all --nproc 6
python prc/adsb/own_flight_features_callsign.py all --nproc 6
python prc/adsb/pos_extract_dep.py all --nproc 6
python prc/adsb/pos_extract_arr_stands.py all --nproc 6
python prc/adsb/lebl_adsb_offblock.py all --nproc 6
python prc/adsb/ampliado_f4_leg_features.py final artifacts/features/lg_final.parquet --nproc 6
python prc/adsb/otimista_features.py final artifacts/features/ox_final.parquet --nproc 6
python prc/adsb/ox_features.py final artifacts/features/ox_LIRF_final.parquet --nproc 6
python prc/adsb/ampliado_f4_make_rt.py --with-2025 --nproc 6
python -m prc.basea.full_chain prepare
python -m prc.basea.full_chain fit --name light70 --threads 8
python -m prc.basea.full_chain fit --name light123 --threads 8
python -m prc.basea.full_chain fit --name light153 --threads 8
python -m prc.basea.full_chain fit --name specialist_single --threads 8
python -m prc.basea.full_chain fit --name specialist_five --threads 8
python -m prc.basea.full_chain fit --name specialist_arrival --threads 8
python -m prc.basea.full_chain fit --name airport_experts --threads 8
python -m prc.basea.full_chain fit --name cat70 --threads 8
python -m prc.basea.full_chain fit --name cat93 --threads 8
python -m prc.basea.full_chain fit --name cat153 --threads 8
python -m prc.basea.full_chain fit --name cat183 --threads 8
python -m prc.basea.assemble
python -m prc.baseb build prc/baseb/configs/h045_h039_mlp_refit.yaml
python -m prc.baseb v35 --write
python -m prc.blend
python prc/layers/camadas_nao_adsb_final.py --pred artifacts/v36_final.parquet --out artifacts/layers/camadas_nao_adsb.parquet
python prc/layers/lebl_chain.py --base artifacts/v36_final.parquet --out artifacts/layers/lebl_chain.parquet
python -m prc.otimista.aplicar_final --pred artifacts/v36_final.parquet --out work/camadaD_final.parquet --months 2,6 --jobs 3
python -m prc.otimista.hbase_prep
python -m prc.otimista.hbase_final --pred artifacts/v36_final.parquet --months 2,6 --jobs 3
python -m prc.layers.compose_layers --e6 artifacts/layers/camadas_nao_adsb_pred.parquet --camada-d work/camadaD_final.parquet
python -m prc.stack.apply_perm --hbase-dir work/hbase/final
python -m prc.espera_rot.dataset
python -m prc.espera_rot.train --threads 8
python -m prc.espera_rot.lirf_dataset
python -m prc.espera_rot.lirf_train
python -m prc.espera_rot.predict --pred artifacts/pre_espera_final.parquet --out artifacts/espera_final.parquet --months 1,2,6,7
python -m prc.espera_rot.lirf_predict --pred artifacts/espera_final.parquet --out artifacts/lirf_final.parquet --months 1,2,6,7
python -m prc.layers.lirf_night_rule --pred artifacts/lirf_final.parquet --out artifacts/night_final.parquet --months 2,6 --dose 0.5
python -m prc.layers.lirf_eco_rule --pred artifacts/night_final.parquet --out artifacts/submission_final.parquet --months 1,2,6,7 --lo-h 6 --hi-h 14 --dose 0.5
python -m prc.stack.nets.seq prc/stack/nets/configs.json --device cpu
python -m prc.stack.mlp.build_unified
python -m prc.stack.mlp.train device=cpu
python -m prc.stack.build --mlp-npz artifacts/oof/mlp_member/final.npz
python -m prc.stack.perm --mlp
python -m prc.stack.mlp_dep --device cpu
python -m prc.stack.candidate --out artifacts/stack_pred.parquet
```

### `scripts/cv.py all`

```
python -m prc.basea.residual_chain_stack_adsb --upto v12 --threads 8
python -m prc.baseb holdout prc/baseb/configs/h045_champion_integration.yaml
python prc/layers/camadas_nao_adsb_final.py --pred artifacts/v36_final.parquet --eval-ltfm --out artifacts/layers/camadas_nao_adsb_eval.parquet
python -m prc.otimista.lmo_final --pred artifacts/v36_final.parquet --out work/lmo_final.csv --months 2,6 --jobs 3
python -m prc.stack.perm --dir artifacts/reference/stack_old --no-nets
python -m prc.espera_rot.cv --threads 8
python -m prc.espera_rot.lirf_cv
```

### `scripts/predict.py all`

```
python -m prc.blend
python -m prc.stack.apply_stack --dir artifacts/reference/stack_old --legacy-months 1,7 --out artifacts/reference/stack_old_pred.parquet
python -m prc.compose --stack artifacts/stack_full_pred.parquet --ref-final artifacts/reference/v78_janjul.parquet --jj-new artifacts/stack_full_pred.parquet --jj-old artifacts/reference/stack_old_pred.parquet --jj-add artifacts/stack_inputs/d_h19.parquet artifacts/stack_inputs/d_pool.parquet artifacts/stack_inputs/d_lirf_fisico.parquet artifacts/stack_inputs/d_row_adjustments_removed.parquet --jj-set artifacts/stack_inputs/set_model_values_on_adjusted_rows.parquet --out artifacts/pre_espera_final.parquet
python -m prc.espera_rot.dataset
python -m prc.espera_rot.train --threads 8
python -m prc.espera_rot.lirf_dataset
python -m prc.espera_rot.lirf_train
python -m prc.espera_rot.predict --pred artifacts/pre_espera_final.parquet --out artifacts/espera_final.parquet --months 1,2,6,7
python -m prc.espera_rot.lirf_predict --pred artifacts/espera_final.parquet --out artifacts/lirf_final.parquet --months 1,2,6,7
python -m prc.layers.lirf_night_rule --pred artifacts/lirf_final.parquet --out artifacts/night_final.parquet --months 2,6 --dose 0.5
python -m prc.layers.lirf_eco_rule --pred artifacts/night_final.parquet --out artifacts/submission_final.parquet --months 1,2,6,7 --lo-h 6 --hi-h 14 --dose 0.5
```

## Validation stages (`scripts/cv.py`)

`basea` fits outside January and July and evaluates on them; `baseb` runs the hold-out configuration; `layers` evaluates the LTFM
layer on 2025; `otimista` runs leave-month-out for February and June; `stack` fits the combiner with nested blocked
cross-validation (blocks of five days, one day purged on each side); `espera` runs the January<->July cross-validation of both
specialists and reports RMSE before/after for doses 0.25, 0.5 and 0.75.

## Post-processing rules

| Rule | Condition | Treatment |
|---|---|---|
| `lirf_night_rule` | LIRF departure, no AOBT, local (Europe/Rome) time of SCHED between 00:00 and 04:00, MVT-SCHED > 1 h, February and June only | 50/50 blend of the model output and `MVT - SCHED` |
| `lirf_eco_rule` | LIRF departure, no AOBT, 6 h <= MVT-SCHED <= 14 h, all four months | 50/50 blend of the model output and `MVT - SCHED` |

`MVT - SCHED` is `MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt` in seconds, computed in UTC. In 2025, among flights where the block
time equals the schedule, `BLOCK - SCHED` has median 0 s, mean -0.39 s and standard deviation 4.3 s, so the quantity is used
without an offset.

## Constants and how they were set

| Constant | Value | How it was set |
|---|---|---|
| Level-1 blend weights | 0.5 / 0.5 | fixed a priori |
| Residual specialist dose and scope | 0.5, nine airports, LIRF excluded | January<->July cross-validation; per-airport dose variants were also evaluated and discarded after leaderboard feedback |
| LIRF specialist dose, filter and cap | 0.5, none, none | January<->July cross-validation; filtered/capped variants were also evaluated and discarded after leaderboard feedback |
| Pooled Huber term alpha | 0.5 | hold-out validation |
| LIRF physical layer dose | 0.5 | hold-out validation on seasonal substitutes |
| Receita-D term dose | 1.149 | fitted on leaderboard scores |
| Blend weights of the reference layer | ridge over earlier outputs, sum of weights 1 | fitted on leaderboard scores |
| Stack safety cut | 1.5x ratio of correction RMS (2026 / OOF), per airport and month; LIRF and LEMD in January fall back | design choice |
| Post-processing rule doses | 0.5 | a priori hedge; the corresponding rule evidence is on small 2025 samples |
