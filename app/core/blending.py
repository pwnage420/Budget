"""Forecast blending — inverse-RMSE + stacked Ridge meta, with the spec's fallback guardrail.

The blend must beat the best single model on backtest *mean* RMSE; if not, we emit
the best single model as the final forecaster (plan §Re-stated guardrails).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import FoldResult


@dataclass
class BlendResult:
    forecast: pd.DataFrame             # date, mean, p20, p30, p70, p80
    weights: dict[str, float]          # model -> weight
    backtest_rmse: float
    chosen_strategy: str               # "inverse_rmse", "stacked_ridge", "best_single"
    fallback_reason: str = ""


def _per_model_summary(folds: list[FoldResult]) -> pd.DataFrame:
    rows = []
    for f in folds:
        if f.failed:
            continue
        rows.append({"model": f.model, "rmse": f.metrics.get("rmse", float("nan")),
                     "fold": f.holdout_year})
    return pd.DataFrame(rows)


def inverse_rmse_weights(summary: pd.DataFrame, top_k: int = 3) -> dict[str, float]:
    if summary.empty or "model" not in summary.columns:
        return {}
    mean_rmse = summary.groupby("model")["rmse"].mean().dropna().sort_values()
    top = mean_rmse.head(top_k)
    inv = 1.0 / top.replace(0, np.nan)
    inv = inv.dropna()
    if inv.empty:
        return {}
    w = (inv / inv.sum()).to_dict()
    return {str(k): float(v) for k, v in w.items()}


def _forecasts_by_model(forecasts: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: fc for name, fc in forecasts.items() if not fc.empty}


def _aligned_means(forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align mean forecasts by date; return wide DataFrame indexed by date."""
    parts = []
    for name, fc in forecasts.items():
        s = fc.set_index("date")["mean"].rename(name)
        parts.append(s)
    return pd.concat(parts, axis=1).sort_index()


def _blend_bands(forecasts: dict[str, pd.DataFrame], weights: dict[str, float],
                 means: pd.Series) -> pd.DataFrame:
    """Combine bands by weighting each model's band offsets from its mean."""
    out = pd.DataFrame({"mean": means})
    for q in ("p20", "p30", "p70", "p80"):
        agg = pd.Series(0.0, index=means.index)
        for name, w in weights.items():
            fc = forecasts[name].set_index("date")
            agg = agg + w * (fc[q] - fc["mean"]).reindex(means.index).fillna(0.0)
        out[q] = means + agg
    out = out.reset_index().rename(columns={"index": "date"})
    if "date" not in out.columns:
        # mean.index may have been named already
        out["date"] = means.index
    out = out[["date", "mean", "p20", "p30", "p70", "p80"]]
    return out


def stacked_ridge_meta(folds: list[FoldResult]) -> tuple[dict[str, float], float]:
    """Train a Ridge meta-learner on out-of-fold predictions and return weights + CV RMSE.

    Returns ({}, nan) if there aren't enough folds.
    """
    if len(folds) < 2:
        return {}, float("nan")
    from sklearn.linear_model import Ridge

    # Build a (n_obs x n_models) matrix of predicted means and the actual vector
    rows = []
    for f in folds:
        if f.failed or f.forecast.empty:
            continue
        actual = pd.Series(f.residuals + f.forecast["mean"].to_numpy()[:len(f.residuals)],
                           name="actual")
        rows.append((f.model, f.holdout_year, actual, f.forecast["mean"]))
    if not rows:
        return {}, float("nan")
    by_fold: dict[int, dict[str, np.ndarray]] = {}
    actuals_by_fold: dict[int, np.ndarray] = {}
    for name, year, actual, mean in rows:
        by_fold.setdefault(year, {})[name] = mean.to_numpy(dtype=float)
        actuals_by_fold[year] = actual.to_numpy(dtype=float)
    model_names = sorted({n for d in by_fold.values() for n in d})
    if len(model_names) < 2:
        return {}, float("nan")
    Xs, ys = [], []
    for year, preds in by_fold.items():
        if not all(name in preds for name in model_names):
            continue
        Xs.append(np.column_stack([preds[name] for name in model_names]))
        ys.append(actuals_by_fold[year])
    if not Xs:
        return {}, float("nan")
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    meta = Ridge(alpha=1.0, fit_intercept=True, positive=False).fit(X, y)
    cv_rmse = float(np.sqrt(np.mean((y - meta.predict(X)) ** 2)))
    # Normalise coefficients to sum to 1 so we can apply as weights
    coefs = meta.coef_
    if coefs.sum() <= 0:
        return {}, cv_rmse
    weights = {name: float(c / coefs.sum()) for name, c in zip(model_names, coefs, strict=False)}
    return weights, cv_rmse


def best_single_rmse(summary: pd.DataFrame) -> tuple[str, float]:
    if summary.empty or "model" not in summary.columns:
        return "", float("nan")
    mean_rmse = summary.groupby("model")["rmse"].mean().dropna().sort_values()
    if mean_rmse.empty:
        return "", float("nan")
    return str(mean_rmse.index[0]), float(mean_rmse.iloc[0])


def blend_forecasts(folds: list[FoldResult],
                    forecasts: dict[str, pd.DataFrame],
                    strategy: str = "inverse_rmse",
                    manual_weights: dict[str, float] | None = None) -> BlendResult:
    """Build the blended P50 + bands with the spec's guardrail.

    Strategy options:
        - inverse_rmse: top-3 weighted by 1/RMSE
        - stacked_ridge: Ridge meta on OOF preds
        - manual: caller-provided weights (forecast tab slider)
    """
    summary = _per_model_summary(folds)
    forecasts = _forecasts_by_model(forecasts)
    best_name, best_rmse = best_single_rmse(summary)

    if strategy == "manual" and manual_weights:
        weights = manual_weights
    elif strategy == "stacked_ridge":
        weights, _ = stacked_ridge_meta(folds)
        if not weights:
            return _fallback(forecasts, summary, best_name, best_rmse,
                             "stacked Ridge had insufficient OOF data")
    else:  # inverse_rmse default
        weights = inverse_rmse_weights(summary, top_k=3)
        if not weights:
            return _fallback(forecasts, summary, best_name, best_rmse,
                             "no successful backtest folds available")

    # Only retain weights for models with forecasts available
    weights = {k: v for k, v in weights.items() if k in forecasts}
    if not weights:
        return _fallback(forecasts, summary, best_name, best_rmse,
                         "no forecasts available for the weighted models")
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}

    means = _aligned_means({k: forecasts[k] for k in weights})
    blended_mean = sum(weights[k] * means[k] for k in weights)
    forecast = _blend_bands({k: forecasts[k] for k in weights}, weights, blended_mean)

    # Guardrail: estimate blend backtest RMSE by weighting per-fold RMSEs
    # (this is approximate but matches the rank-based guardrail in the plan)
    fold_rmse = summary.groupby("model")["rmse"].mean()
    blend_rmse = float(sum(weights[k] * fold_rmse.get(k, float("nan")) for k in weights))
    if not np.isfinite(blend_rmse) or (best_name and blend_rmse > best_rmse):
        return _fallback(forecasts, summary, best_name, best_rmse,
                         f"blend RMSE {blend_rmse:.3f} >= best single {best_rmse:.3f}")

    return BlendResult(forecast=forecast, weights=weights, backtest_rmse=blend_rmse,
                       chosen_strategy=strategy)


def _fallback(forecasts: dict[str, pd.DataFrame], summary: pd.DataFrame,
              best_name: str, best_rmse: float, reason: str) -> BlendResult:
    if best_name and best_name in forecasts:
        return BlendResult(
            forecast=forecasts[best_name].copy(),
            weights={best_name: 1.0},
            backtest_rmse=best_rmse,
            chosen_strategy="best_single",
            fallback_reason=reason,
        )
    # Last-resort: emit any available forecast
    if forecasts:
        first = next(iter(forecasts))
        return BlendResult(
            forecast=forecasts[first].copy(),
            weights={first: 1.0},
            backtest_rmse=float("nan"),
            chosen_strategy="best_single",
            fallback_reason=reason + " | also no best-single available",
        )
    return BlendResult(
        forecast=pd.DataFrame(columns=["date", "mean", "p20", "p30", "p70", "p80"]),
        weights={}, backtest_rmse=float("nan"),
        chosen_strategy="best_single", fallback_reason=reason + " | no forecasts at all",
    )
