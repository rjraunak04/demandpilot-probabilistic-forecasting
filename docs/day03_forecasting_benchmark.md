# Day 03 — Forecasting Benchmark and Model Selection

## Objective

Day 03 establishes a leakage-aware point-forecasting benchmark before
probabilistic forecasting and inventory optimization are introduced.

## Evaluation policy

The model-ready SKU-day feature table is split chronologically:

| Partition | Date range | Rows |
| --- | --- | ---: |
| Train | through 2011-10-14 | 2,018,615 |
| Validation | 2011-10-15 to 2011-11-11 | 81,092 |
| Locked test | 2011-11-12 to 2011-12-09 | 70,360 |

The locked test partition is not used for hyperparameter tuning or champion
selection. WAPE is the primary metric because the demand panel contains many
zero-demand SKU-days. MAE, RMSE, and sMAPE are retained as diagnostics.

## Baselines

The validation benchmark established two deterministic references:

| Model | WAPE |
| --- | ---: |
| Seasonal naive (lag 7) | 1.224818 |
| Naive (lag 1) | 1.363237 |

Across four 7-day rolling-origin folds, seasonal naive achieved mean WAPE
1.234057 (standard deviation 0.132476), making it the baseline to beat.

## Gradient boosting

Untuned validation results:

| Model | WAPE | MAE | RMSE | sMAPE |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.853492 | 6.416482 | 39.944887 | 1.573577 |
| LightGBM | 0.863008 | 6.488021 | 40.431170 | 0.764024 |

Both gradient-boosting models materially outperform the seasonal baseline on
the primary WAPE metric.

## Time-aware tuning

Optuna uses a bounded search (8 trials per model). The objective is evaluated
only inside the training partition using a chronological 180-day fit window
followed by a 7-day tuning window. Official validation and locked test data are
not part of the tuning objective.

Best internal tuning WAPE:

- LightGBM: 0.843780
- XGBoost: 0.855672

## Champion selection

The tuned parameter sets are retrained on the full training partition and
compared on the official validation partition.

| Model | Validation WAPE |
| --- | ---: |
| **LightGBM** | **0.860410** |
| XGBoost | 0.861949 |
| Seasonal naive | 1.224818 |
| Naive lag 1 | 1.363237 |

**Champion: LightGBM.**

Relative WAPE improvement over seasonal naive:

`(1.224818 - 0.860410) / 1.224818 = 29.75%`.

The higher XGBoost sMAPE is reported rather than hidden; sMAPE behaves
differently from WAPE on this sparse, zero-heavy SKU-day panel. Champion
selection follows the predeclared primary metric, WAPE.

## Reproducibility

Core commands:

```powershell
python scripts\benchmark_baselines.py
python scripts\run_baseline_backtest.py
python scripts\benchmark_boosting.py
python scripts\tune_boosting.py
python scripts\select_champion.py
```

Generated outputs are written to `reports/modeling/` and intentionally ignored
by Git. They can be reconstructed from the versioned code and processed feature
artifact.

## Day-03 boundary

Day 03 selects the deterministic point-forecasting champion. Probabilistic
P10/P50/P90 forecasts and inventory decision logic belong to Day 04.
