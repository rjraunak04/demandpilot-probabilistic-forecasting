# Chronological Forecast Evaluation

DemandPilot evaluates forecasting models using time-ordered partitions rather
than random row-wise train/test splits.

## Forecast Horizon

The primary operational forecast horizon is 7 days.

## Partitions

- Training: through 2011-10-14
- Validation: 2011-10-15 through 2011-11-11
- Locked test: 2011-11-12 through 2011-12-09

Validation and test each contain 28 calendar days, providing four 7-day
evaluation periods.

## Historical Eligibility

Rows must have complete 28-day historical lag and rolling features before
entering model evaluation. Early-history rows remain in the feature artifact
but are excluded from model fitting and scoring.

## Leakage Policy

Training observations always precede validation observations, and validation
observations always precede locked-test observations.

The locked test set is not used for hyperparameter tuning or model selection.

## SKU Availability

Demand is evaluated only over SKU-days represented by the Day-02 lifetime
panel. DemandPilot does not create artificial demand observations before a
SKU's first observed sale or after its last observed sale.

This avoids interpreting unknown product availability or discontinuation as
known zero demand.

## Final-Period Caveat

The number of eligible SKUs declines toward the dataset endpoint. This is
reported transparently rather than corrected by synthetic post-lifetime zeros.

Aggregate metrics must therefore be interpreted together with SKU-day
coverage.