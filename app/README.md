# Goodness Grown — Tomato Price Forecaster

A Python desktop application that forecasts and tracks weekly retail tomato prices
($/kg) for Goodness Grown, the 20-hectare high-tech glasshouse in Tongala, Victoria.
It replaces the Excel-based budget model with a panel of statistical and machine-
learning forecasters, calibrated probability bands, walk-forward backtesting, live
financial-year actuals tracking, and one-click Excel + PDF board exports.

---

## 1. What this app does

Replaces the Excel budget model with a panel of statistical and machine-learning
forecasters that produce a calibrated probability range for weekly retail tomato
$/kg across the AU financial year (July–June), and tracks live variance against
actuals as the year progresses.

## 2. Why move off Excel

- **Exogenous regressors** — weather, promo calendar, supply index, fuel — that
  Excel cannot ingest cleanly are first-class inputs here.
- **Probability-distribution models** (Bayesian Structural Time Series, N-BEATS,
  Temporal Fusion Transformer) give full distributions, not just mean + bands.
- **Auto-backtesting** ranks every model on out-of-sample RMSE rather than relying
  on an eyeballed in-sample fit.
- **Audit trail** — every forecast run is timestamped, parameter-logged, and
  reproducible (fixed seeds, captured library versions).
- **Calibrated bands** — historical P20-P80 band coverage is checked to hit ≈60%
  on holdout actuals, not just assumed.

## 3. Quickstart

### End user (Windows .exe)

Download the latest `tomato-forecaster-lite.exe`, double-click. Click *Use sample
data* on the Data tab, then *Run All* on the Models tab. The Forecast tab shows
the blended P50 with P20-P80 amber bands; *Export PDF…* writes the board pack.

### Analyst (Python)

```bash
git clone https://github.com/pwnage420/budget.git
cd budget
pip install -r app/requirements.txt
python -m app.main
```

For the full panel (Prophet, Auto-SARIMA, N-BEATS, TFT, BSTS):

```bash
pip install -r app/requirements-full.txt
```

### Headless / CI

```bash
python -m app.demo --seed 42 --out ./out
```

Runs the full golden path with no GUI: loads sample data, fits the panel,
backtests, blends, writes Excel + PDF, and asserts monotone bands + correct
sheet/page counts. Used by the CI workflow.

## 4. The data contract

Two layouts are accepted; pick whichever suits your existing process:

**Wide** (matches the v4 Excel model):

```
Week,2021,2022,2023,2024,2025
1,3.80,3.50,3.05,3.50,4.05
2,2.95,3.20,3.05,3.30,3.10
...
```

**Long**:

```
Date,Price
2021-01-04,3.80
2021-01-11,2.95
...
```

Validation rules:
- 52 weeks per year (53 ISO weeks handled with a configurable rule).
- All prices must be strictly positive ($/kg).
- Files larger than 50 MB are rejected (security: zip/XML bomb mitigation).
- Macro-enabled Excel (.xlsm, .xltm) is rejected outright.
- Outliers are flagged via the robust MAD form `|y - median| > 3·1.4826·MAD`
  per week-of-year. Flagged rows are **never auto-corrected** — they highlight
  amber in the Data tab preview with an include/exclude checkbox.

Regressor files must have `year` and `week` columns plus one or more numeric
regressor columns. Built-in sample regressors include `max_temp_c`, `rainfall_mm`,
`promo_flag`, `public_holiday_week`, `school_holiday_week`, `fuel_index`.

AU Financial Year defaults to July–June. Week 1 of FY is the ISO week containing
the first Monday on or after the FY-start date. Configurable in Settings.

## 5. How the forecasting works — narrative overview

Four-stage pipeline:

1. **Load + validate** — wide or long CSV/Excel is ingested, weeks-per-year and
   gaps are checked, prices are validated as strictly positive, outliers are
   robustly flagged via MAD.
2. **Fit panel of models on history** — every model in the panel is fitted on
   the full history. Statistical (Naive, Seasonal+Trend, Holt-Winters, SARIMA),
   ML (Random Forest, XGBoost, LightGBM), Prophet, and optionally Auto-SARIMA /
   N-BEATS / TFT / Bayesian Structural TS when the Advanced toggle is on.
3. **Walk-forward backtest** — expanding-window cross-validation on the most
   recent holdout year(s). Each model produces a 52-week forecast for the
   holdout year; metrics (RMSE, MAE, MAPE, MASE, bias, P20-P80 coverage) are
   computed and the panel is ranked by mean RMSE.
4. **Blend** — the top-3 models are weighted by inverse-RMSE. Probability bands
   come from the residual standard deviation pooled across folds and nearby
   horizon steps, then rescaled so the historical P20-P80 band covers ≈60% of
   holdout actuals. If the blended model doesn't beat the best single model on
   backtest, the best single is emitted instead (spec guardrail).

## 6. Each model in plain English

- **Naive (last-year same week)** — predicts price equal to the same week last
  year. Surprisingly hard to beat on smooth seasonal series.
- **Seasonal Avg + Trend** — average price across years for that week, plus a
  linear OLS trend on the de-seasonalised residual. Excel-tier baseline.
- **Holt-Winters ETS** — exponential smoothing with level, trend, and 52-week
  seasonal terms. Robust default; copes well when the seasonal shape is stable.
- **SARIMA(1,0,1)(1,1,1)[52]** — autoregressive moving average with a seasonal
  difference. Cited limitation: with only 5 yearly cycles, parameters are weakly
  identified; warnings are common. See §10 Limitations.
- **Auto-SARIMA** (Advanced) — pmdarima's stepwise search across orders.
- **Prophet** (Advanced) — Meta's additive decomposition `y = g + s + h + ε`,
  with yearly seasonality and optional exogenous regressors via `add_regressor`.
- **Random Forest / XGBoost / LightGBM** — gradient-boosted trees trained on
  engineered features (lags, rolling means/stds, Fourier seasonal terms, calendar
  + trend). Forecast recursively with predicted lags feeding back into features.
  LightGBM won the M5 Forecasting Competition on retail data of this kind.
- **Bayesian Structural TS** (Advanced) — PyMC implementation of a local-level +
  seasonal + AR(1) state-space model. Returns full posterior samples → real
  probability distributions, not just mean ± band.
- **N-BEATS / TFT** (Advanced) — darts/PyTorch deep models. Included for
  completeness; with 260 observations they will overfit unless you accept the
  small networks + heavy regularisation defaults. See §10 Limitations.
- **Inverse-RMSE Blend** — `wᵢ = (1/RMSEᵢ) / Σⱼ(1/RMSEⱼ)` over the top 3 models.
- **Stacked Ridge** — meta-learner fitted on out-of-fold predictions of each base
  model, regularised to discourage over-reliance on any one input.

## 7. The maths

### Error metrics

- **RMSE** = √(mean((ŷ − y)²))
- **MAE** = mean(|ŷ − y|)
- **MAPE** = mean(|(ŷ − y) / y|) — guarded against zero actuals
- **sMAPE** = mean(|ŷ − y| / ((|ŷ| + |y|) / 2))
- **MASE** = MAE / mean(|yₜ − yₜ₋₅₂|) (seasonal-naive scale, lag-52 for weekly)
- **Bias** = mean(ŷ − y) — positive means systematic over-forecast
- **Coverage** = fraction of actuals inside [P20, P80]; target ≈ 0.60
- **CRPS** = sample-based Continuous Ranked Probability Score (lower is better)

### Feature engineering for tree models

- Lags `y_{t−1}, y_{t−2}, y_{t−4}, y_{t−52}`
- Rolling mean and std over 4, 13, 52 week windows (shifted to avoid leakage)
- Calendar: week-of-year, month, quarter, year
- Fourier: `sin(2πk·w/52), cos(2πk·w/52)` for k = 1, 2, 3
- Linear trend `t = 0, 1, 2, …`

### SARIMA(p,d,q)(P,D,Q)[s]

State-space form with one weekly autoregressive term, one weekly moving-average
term, one seasonal AR, one seasonal MA, and one seasonal difference. We use
(p=1, d=0, q=1)(P=1, D=1, Q=1)[s=52] as a baseline and `pmdarima.auto_arima` for
search (capped at p,q ≤ 2; P,Q ≤ 1).

### Holt-Winters

Level/trend/seasonal recursions: ℓₜ = α(yₜ − sₜ₋ₛ) + (1−α)(ℓₜ₋₁ + bₜ₋₁);
bₜ = β(ℓₜ − ℓₜ₋₁) + (1−β)bₜ₋₁; sₜ = γ(yₜ − ℓₜ) + (1−γ)sₜ₋ₛ.

### Prophet

`y(t) = g(t) + s(t) + h(t) + ε`, with `g` piecewise-linear or logistic growth, `s`
the yearly seasonality (Fourier series), `h` user-specified holidays, and
regressors entering via `add_regressor`.

### Walk-forward backtest

```
for hy in last K years:
    train = data[year < hy]
    test  = data[year == hy]
    fit(model, train)
    yhat = predict(model, len(test))
    record metrics(yhat, test)
```

### Inverse-RMSE blend

`wᵢ = (1/RMSEᵢ) / Σⱼ(1/RMSEⱼ)` over the top-3 models; blended P50 = Σᵢ wᵢ·ŷᵢ.

### Stacked Ridge meta-learner

Training inputs are the out-of-fold predictions of each base model across all
backtest folds; meta-coefficients minimise `‖Xβ − y‖² + λ‖β‖²`.

### Probability bands

`P_q(h) = ŷ(h) + z_q · σ(h)`, where `σ(h)` is the residual standard deviation at
horizon `h`, pooled across folds and a sliding ±4-week horizon window. After
parametric bands are computed, a single multiplicative correction `c` rescales σ
so that historical P20-P80 coverage hits 60% ± 5% across the backtest folds —
the spec's calibration requirement made operational. With ≥ 8 backtest folds the
quantile path uses empirical per-horizon residuals instead.

## 8. Glossary

| Abbrev | Definition |
|---|---|
| AR | Autoregressive — value depends on its own past values |
| ARIMA | Autoregressive Integrated Moving Average |
| Backtest | Evaluating a model on historical data it didn't see during training |
| Bias | Mean signed forecast error |
| BoM | Bureau of Meteorology (AU weather agency) |
| BSTS | Bayesian Structural Time Series |
| Coverage | Fraction of actuals inside a probability band |
| CRPS | Continuous Ranked Probability Score (proper scoring rule) |
| CV | Cross-validation |
| ETS | Exponential Smoothing State Space model |
| FY | Financial Year (Jul–Jun in Australia) |
| GM | General Manager |
| HMAC | Hash-based Message Authentication Code (cache signing) |
| KPI | Key Performance Indicator |
| LGBM / LightGBM | Light Gradient Boosting Machine |
| LY / TY | Last Year / This Year |
| MA | Moving Average |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| MAD | Median Absolute Deviation |
| MASE | Mean Absolute Scaled Error |
| N-BEATS | Neural Basis Expansion Analysis for Time Series |
| OOS | Out-of-sample |
| P20 / P50 / P80 | 20th / 50th / 80th percentile of forecast distribution |
| PIT | Probability Integral Transform (calibration diagnostic) |
| Prophet | Additive decomposition forecasting library (Meta) |
| QSS | Qt Style Sheets (PyQt theming) |
| RF | Random Forest |
| RMSE | Root Mean Squared Error |
| SARIMA | Seasonal ARIMA |
| SHAP | SHapley Additive exPlanations |
| sMAPE | Symmetric MAPE |
| TFT | Temporal Fusion Transformer |
| XGB / XGBoost | eXtreme Gradient Boosting |

## 9. Reproducibility

- Seeds are set once at process start and again inside every model worker
  (`app/core/seeding.py`). NumPy, Python `random`, PyTorch, and tree-model RNGs
  are all covered.
- Every run writes an audit record to `~/.tomato-forecaster/runs/<iso-ts>.json`
  containing the SHA-256 of the input data, the model list and weights, the
  blend backtest RMSE, library versions, and the seed. Records contain the data
  *hash*, not raw data values (security).
- `python -m app.demo --reproducibility-check` runs the demo twice and asserts
  byte-identical Excel exports under the same seed.

## 10. Limitations

- **Sample size** — 5 years × 52 weeks = 260 observations. Deep learning models
  (N-BEATS, TFT) typically need thousands; they're included for spec
  completeness but lean on early stopping + small networks. Don't expect them
  to consistently beat LightGBM at this size.
- **SARIMA with seasonal period 52** is borderline at 5 cycles — Hyndman &
  Athanasopoulos (FPP3 §12.1) recommend Fourier-term dynamic regression in this
  regime, which is what Prophet and our tree models with Fourier features do.
- **Bands assume residual distributions are roughly stationary across horizons**
  — we recalibrate on every backtest but the parametric assumption may fail in
  regime changes.
- **Forecasting with regressors requires regressor values for the future weeks**
  — user-controllable regressors (promo, school/public holidays) are user-input;
  weather extrapolates via long-run climatology by default and the chart badges
  the assumption.

## 11. Troubleshooting / FAQ

- **"Data did not pass validation"** — check the message for the specific reason:
  non-positive prices, fewer than 52 weeks per year, unknown layout. The file is
  never modified; fix the source and reload.
- **"Library not installed"** in the Models tab — that model needs an Advanced
  dependency; either toggle Advanced off in Settings or install
  `requirements-full.txt`.
- **A model row is grey with a red dot** — it failed; click the row to see the
  plain-English failure reason. Other models keep running.
- **Reset config** — delete `~/.tomato-forecaster/config.json` and relaunch.
- **Cache lives at** `~/.tomato-forecaster/cache/`; safe to delete.

## 12. Methodology summary for the board

The forecast is produced by fitting a panel of statistical and machine-learning
models to historical weekly retail prices, ranking each by out-of-sample error
on a held-out year, blending the top three by inverse-error weighting, and
calibrating the probability bands so they historically cover ≈60% of holdout
actuals. The methodology follows standard practice in time-series forecasting
(Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* 3e, 2021),
extended with techniques drawn from competition-winning approaches in M4 and M5
(Makridakis et al., 2022) — gradient-boosted trees on engineered features for
the point forecast, calibrated parametric bands from residual variance, and an
explicit guardrail that falls back to the best single model if the blend fails
to improve on backtest.

The audit log captures every model parameter and the data hash so any historical
forecast can be reproduced exactly from its record.

## 13. References

- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
  Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American
  Statistician*, 72(1), 37–45.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  *KDD '16*.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision
  tree. *NeurIPS '17*.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
  competition: results, findings, and conclusions. *International Journal of
  Forecasting*, 38(4), 1346–1364.
- Oreshkin, B. N., et al. (2020). N-BEATS: Neural basis expansion analysis for
  interpretable time series forecasting. *ICLR '20*.
- Lim, B., et al. (2021). Temporal Fusion Transformers for interpretable
  multi-horizon time series forecasting. *International Journal of Forecasting*,
  37(4), 1748–1764.
- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts.
  *Operational Research Quarterly*, 20(4), 451–468.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–
  259.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
  prediction, and estimation. *JASA*, 102(477), 359–378.
- Diebold, F. X., Gunther, T. A., & Tay, A. S. (1998). Evaluating density
  forecasts. *International Economic Review*, 39(4), 863–883.
- Bontempi, G., Ben Taieb, S., & Le Borgne, Y. (2013). Machine learning
  strategies for time series forecasting. *LNBIP* 138.
- Leys, C., et al. (2013). Detecting outliers: Do not use standard deviation
  around the mean, use absolute deviation around the median. *J. Exp. Soc.
  Psychology*, 49(4), 764–766.

## 14. Project metadata

- Authors: Goodness Grown analytics team + Claude Code-on-the-web.
- Licence: internal use; not for redistribution without permission.
- Repo: `pwnage420/budget` on the Code-on-the-web GitHub.

## Known deviations from spec

- The `tomato-forecaster-full` PyInstaller bundle (with PyTorch + darts + PyMC)
  will exceed the 250 MB cap — recommend the `lite` build for the GM, `full` for
  analyst use.
- The 3-minute panel-runtime target is met by `lite`. With the Advanced toggle
  on, N-BEATS + TFT + BSTS on a no-GPU laptop will exceed 3 min; the toggle
  text says so.
- The v4 Excel layout is not in the repo; the export ships with the nine spec'd
  sheets in order and a clean default layout per sheet. Once the v4 file is
  shared, the per-sheet layout can be re-aligned without restructuring the
  export module.
