"""External regressor ingestion + alignment + correlation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loader import DataLoadError, _parse_dataframe, _read_bytes

REQUIRED_COLS = {"year", "week"}


def load_regressors(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    raw = _read_bytes(p)
    df = _parse_dataframe(raw, p.suffix)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if not REQUIRED_COLS.issubset(lower):
        raise DataLoadError(
            "Regressor file must include 'year' and 'week' columns (and one or more "
            "regressor columns alongside)."
        )
    df = df.rename(columns={lower["year"]: "year", lower["week"]: "week"})
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["date"] = df.apply(
        lambda r: pd.Timestamp.fromisocalendar(int(r["year"]), int(r["week"]), 1), axis=1,
    )
    cols = ["date", "year", "week"] + [c for c in df.columns
                                        if c not in {"date", "year", "week"}]
    return df[cols].sort_values(["year", "week"]).reset_index(drop=True)


def correlation_with_price(prices: pd.DataFrame,
                           regressors: pd.DataFrame) -> pd.Series:
    """Pearson correlation between price and each numeric regressor column."""
    merged = prices[["year", "week", "price"]].merge(
        regressors, on=["year", "week"], how="inner"
    )
    numeric_cols = [c for c in merged.columns
                    if c not in {"year", "week", "date", "price"}
                    and pd.api.types.is_numeric_dtype(merged[c])]
    corrs = {col: float(merged["price"].corr(merged[col])) for col in numeric_cols}
    return pd.Series(corrs, name="correlation").sort_values(ascending=False)


def climatology_future(history: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Default future regressor values via 5-year climatology (mean per week-of-year).

    Used when the user toggles a regressor on but hasn't supplied future values for the
    forecast horizon — see plan §Open ambiguities, item 3.
    """
    means = history.drop(columns=["date", "year"], errors="ignore").groupby("week").mean(numeric_only=True)
    last_date = pd.Timestamp(history["date"].max())
    rows = []
    for h in range(1, horizon + 1):
        d = last_date + pd.Timedelta(weeks=h)
        iso = d.isocalendar()
        row = {"date": d, "year": int(iso.year), "week": int(iso.week)}
        for col in means.columns:
            row[col] = float(means.loc[int(iso.week), col]) if int(iso.week) in means.index else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
