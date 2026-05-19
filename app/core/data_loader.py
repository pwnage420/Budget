"""CSV / Excel ingestion + validation.

Accepts either wide (Week,2021,...,2025) or long (Date,Price) layouts. Validates
weeks-per-year, gaps, non-positive prices, and flags outliers using the robust MAD
form (see README §7 maths and Evidence section of the plan). Never auto-corrects —
flags only.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB hard cap (security: zip / XML bombs)
WEEKS_PER_YEAR = 52
MAD_K = 1.4826  # constant making MAD a consistent estimator of std under normality
OUTLIER_THRESHOLD = 3.0


@dataclass
class ValidationReport:
    years: list[int] = field(default_factory=list)
    weeks_per_year: dict[int, int] = field(default_factory=dict)
    missing: list[tuple[int, int]] = field(default_factory=list)  # (year, week)
    outliers: list[tuple[int, int, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = [
            f"Years detected: {', '.join(str(y) for y in self.years) or 'none'}",
            "Weeks per year: " + (
                ", ".join(f"{y}:{n}" for y, n in self.weeks_per_year.items()) or "n/a"
            ),
            f"Missing weeks: {len(self.missing)}",
            f"Outliers flagged: {len(self.outliers)}",
        ]
        if self.warnings:
            parts.append("Warnings: " + " | ".join(self.warnings))
        if self.errors:
            parts.append("Errors: " + " | ".join(self.errors))
        return "\n".join(parts)


@dataclass
class LoadedSeries:
    """Long-format series indexed by ISO date (Monday-of-week)."""

    df: pd.DataFrame  # columns: date, year, week, price
    report: ValidationReport


class DataLoadError(ValueError):
    """Plain-English error class — raised at the GUI boundary."""


def _read_bytes(path: str | Path) -> bytes:
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File not found: {p}")
    size = p.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise DataLoadError(
            f"File too large ({size / 1e6:.1f} MB > 50 MB cap). Trim before uploading."
        )
    if p.suffix.lower() in {".xlsm", ".xltm"}:
        raise DataLoadError("Macro-enabled Excel files (.xlsm/.xltm) are not accepted.")
    return p.read_bytes()


def _parse_dataframe(raw: bytes, suffix: str) -> pd.DataFrame:
    suffix = suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(io.BytesIO(raw))
    if suffix in {".xlsx", ".xls"}:
        # openpyxl read_only is safer against XML/zip-bomb attacks
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    raise DataLoadError(f"Unsupported file type: {suffix}. Use CSV or XLSX.")


def _detect_layout(df: pd.DataFrame) -> str:
    cols = [str(c).strip().lower() for c in df.columns]
    if {"date", "price"}.issubset(cols):
        return "long"
    if cols and cols[0] in {"week", "wk", "week_of_year"} and len(cols) >= 2:
        return "wide"
    if "date" in cols:
        return "long"
    raise DataLoadError(
        "Could not detect layout. Expected either 'Week,2021,...,2025' (wide) or "
        "'Date,Price' (long). Check the first row of your file."
    )


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    week_col = df.columns[0]
    year_cols = []
    for c in df.columns[1:]:
        try:
            year_cols.append((c, int(c)))
        except ValueError:
            continue
    if not year_cols:
        raise DataLoadError("Wide layout has no year columns (expected '2021', '2022', ...).")
    long = df.melt(id_vars=[week_col], value_vars=[c for c, _ in year_cols],
                   var_name="year", value_name="price")
    long.rename(columns={week_col: "week"}, inplace=True)
    long["year"] = long["year"].astype(int)
    long["week"] = long["week"].astype(int)
    long["date"] = long.apply(lambda r: _iso_week_monday(int(r["year"]), int(r["week"])), axis=1)
    return long[["date", "year", "week", "price"]].sort_values(["year", "week"]).reset_index(drop=True)


def _iso_week_monday(year: int, week: int) -> pd.Timestamp:
    """Monday of ISO week (year, week)."""
    # Use ISO calendar — `%G-W%V-%u` is the canonical format
    return pd.Timestamp.fromisocalendar(year, max(1, min(week, 52)), 1)


def _long_normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise DataLoadError("Some Date values could not be parsed. Use YYYY-MM-DD.")
    iso = df["date"].dt.isocalendar()
    df["year"] = iso["year"].astype(int)
    df["week"] = iso["week"].astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df[["date", "year", "week", "price"]].sort_values(["year", "week"]).reset_index(drop=True)


def _validate(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    if df.empty:
        report.errors.append("No data rows found.")
        return report

    if df["price"].isna().any():
        n = int(df["price"].isna().sum())
        report.warnings.append(f"{n} row(s) have missing prices — masked from training.")
    if (df["price"].fillna(1) <= 0).any():
        report.errors.append("Prices must be strictly positive ($/kg).")

    years = sorted(df["year"].unique().tolist())
    report.years = years
    for y in years:
        sub = df[df["year"] == y]
        report.weeks_per_year[y] = int(sub["week"].nunique())
        present_weeks = set(int(w) for w in sub["week"].unique())
        expected = set(range(1, WEEKS_PER_YEAR + 1))
        for w in sorted(expected - present_weeks):
            report.missing.append((y, w))

    # Robust outlier flagging: per week-of-year, MAD-based 3-sigma equivalent.
    # When MAD is exactly 0 (e.g. constant history with one extreme value) we fall
    # back to std-based detection so the single outlier still gets flagged.
    for w in range(1, WEEKS_PER_YEAR + 1):
        sub = df[df["week"] == w]
        prices = sub["price"].dropna().to_numpy(dtype=float)
        if len(prices) < 3:
            continue
        med = float(np.median(prices))
        mad = float(np.median(np.abs(prices - med)))
        if mad > 0:
            threshold = OUTLIER_THRESHOLD * MAD_K * mad
            for _, row in sub.iterrows():
                if pd.isna(row["price"]):
                    continue
                if abs(float(row["price"]) - med) > threshold:
                    report.outliers.append((int(row["year"]), int(row["week"]), float(row["price"])))
        else:
            # MAD = 0 means almost all observations equal the median. Any
            # observation that differs at all is a candidate outlier by definition;
            # a single inflating value otherwise makes the std-based threshold too
            # lax to detect itself.
            for _, row in sub.iterrows():
                if pd.isna(row["price"]):
                    continue
                if not np.isclose(float(row["price"]), med, atol=1e-6):
                    report.outliers.append((int(row["year"]), int(row["week"]), float(row["price"])))
    return report


def load_series(path: str | Path) -> LoadedSeries:
    """Load and validate a weekly price file. Raises DataLoadError with a user-readable msg."""
    p = Path(path)
    raw = _read_bytes(p)
    df = _parse_dataframe(raw, p.suffix)
    layout = _detect_layout(df)
    long = _wide_to_long(df) if layout == "wide" else _long_normalise(df)
    report = _validate(long)
    if not report.ok:
        raise DataLoadError("Data did not pass validation: " + " | ".join(report.errors))
    return LoadedSeries(df=long, report=report)


def load_series_from_dataframe(df: pd.DataFrame) -> LoadedSeries:
    """Programmatic loader (used by sample data + tests)."""
    cols = [str(c).strip().lower() for c in df.columns]
    if {"date", "price"}.issubset(cols):
        long = _long_normalise(df)
    elif cols and cols[0] in {"week", "wk", "week_of_year"}:
        long = _wide_to_long(df)
    else:
        raise DataLoadError("Unrecognised DataFrame layout.")
    report = _validate(long)
    if not report.ok:
        raise DataLoadError("Data did not pass validation: " + " | ".join(report.errors))
    return LoadedSeries(df=long, report=report)
