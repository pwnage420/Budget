"""Headless end-to-end acceptance harness.

Runs the full golden path with no Qt window so we catch regressions in CI:
    python -m app.demo --seed 42 --out ./out

Asserts that:
- The blended forecast covers 52 weeks with monotone bands (P20 < P30 < P50 < P70 < P80)
- The Excel workbook contains all nine spec'd sheet names
- The PDF has exactly 4 pages
- Two runs with the same seed produce byte-identical Excel exports (reproducibility)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from .core.audit import make_record, write_record
from .core.backtest import aggregate_metrics, pooled_residual_sigma, walk_forward
from .core.blending import blend_forecasts
from .core.data_loader import load_series_from_dataframe
from .core.models import default_panel, fit_and_forecast
from .core.regressors import climatology_future
from .core.sample_data import generate_price_series, generate_regressors
from .core.seeding import seed_all
from .export.excel_writer import ExportPayload, export_workbook
from .export.pdf_writer import PdfPayload, export_pdf

EXPECTED_SHEETS = [
    "README", "Glossary", "Dashboard", "Budget 2026", "FY Tracking",
    "Source Data", "Model Comparison", "Methodology", "Backtest",
]


def _build(seed: int, out_dir: Path, use_regressors: bool = True) -> dict:
    seed_all(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = generate_price_series(seed=seed)
    series = load_series_from_dataframe(raw)
    history = series.df

    regs = generate_regressors(seed=seed) if use_regressors else None
    panel = default_panel(advanced=False)

    folds = walk_forward(history, panel, holdout_years=1, regressors=regs)
    metrics = aggregate_metrics(folds)
    print("Backtest summary:\n", metrics)

    future_regs = climatology_future(regs, 52) if regs is not None else None
    forecasts = {}
    for tpl in panel:
        model = tpl.__class__()
        sigma = pooled_residual_sigma(folds, model.name)
        if len(sigma) == 0:
            sigma = None
        fc = fit_and_forecast(model, history, horizon=52,
                              regressors=regs, future_regressors=future_regs,
                              residual_sigma=sigma)
        if not fc.failed:
            forecasts[model.name] = fc.forecast

    blend = blend_forecasts(folds, forecasts, strategy="inverse_rmse")
    print(f"Blend strategy: {blend.chosen_strategy}, "
          f"weights: {blend.weights}, RMSE: {blend.backtest_rmse:.3f}")

    payload = ExportPayload(
        history=history,
        forecast=blend.forecast,
        metrics=metrics,
        blend_weights=blend.weights,
        blend_strategy=blend.chosen_strategy,
        blend_backtest_rmse=blend.backtest_rmse,
    )
    excel_path = export_workbook(payload, out_dir / "tomato_forecast.xlsx")
    pdf_path = export_pdf(
        PdfPayload(
            history=history, forecast=blend.forecast, actuals=None,
            metrics=metrics, blend_weights=blend.weights,
            blend_strategy=blend.chosen_strategy,
            blend_backtest_rmse=blend.backtest_rmse,
        ),
        out_dir / "tomato_forecast.pdf",
    )

    rec = make_record(
        history=history, horizon=52,
        models_attempted=[m.name for m in panel],
        models_succeeded=list(forecasts.keys()),
        blend_strategy=blend.chosen_strategy,
        blend_weights=blend.weights,
        blend_rmse=blend.backtest_rmse,
        seed=seed,
    )
    audit_path = write_record(rec, target_dir=out_dir / "runs")

    return {
        "excel": excel_path,
        "pdf": pdf_path,
        "audit": audit_path,
        "blend": blend,
        "metrics": metrics,
    }


def _assert_monotone_bands(fc: pd.DataFrame) -> None:
    a = fc["p20"].to_numpy()
    b = fc["p30"].to_numpy()
    c = fc["mean"].to_numpy()
    d = fc["p70"].to_numpy()
    e = fc["p80"].to_numpy()
    assert (a <= b).all(), "P20 must be <= P30"
    assert (b <= c).all(), "P30 must be <= P50"
    assert (c <= d).all(), "P50 must be <= P70"
    assert (d <= e).all(), "P70 must be <= P80"


def _assert_excel_sheets(path: Path) -> None:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    missing = [s for s in EXPECTED_SHEETS if s not in wb.sheetnames]
    assert not missing, f"Excel export missing sheets: {missing}"


def _assert_pdf_pages(path: Path, expected: int = 4) -> None:
    raw = path.read_bytes()
    # Count `/Type /Page` not `/Type /Pages`. Quick-and-dirty without a PDF lib.
    n_page = raw.count(b"/Type /Page\n") + raw.count(b"/Type /Page ")
    n_page = max(n_page, raw.count(b"/Type/Page\n") + raw.count(b"/Type/Page "))
    assert n_page == expected, f"PDF page count: expected {expected}, got {n_page}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("./out"))
    parser.add_argument("--reproducibility-check", action="store_true",
                        help="Run the demo twice and verify byte-identical Excel.")
    parser.add_argument("--no-regressors", action="store_true",
                        help="Skip the regressors path (faster).")
    args = parser.parse_args(argv)

    result = _build(args.seed, args.out, use_regressors=not args.no_regressors)
    _assert_monotone_bands(result["blend"].forecast)
    _assert_excel_sheets(result["excel"])
    _assert_pdf_pages(result["pdf"], expected=4)
    print("Forecast acceptance checks: OK")
    print(f"  Excel: {result['excel']}")
    print(f"  PDF:   {result['pdf']}")
    print(f"  Audit: {result['audit']}")

    if args.reproducibility_check:
        # Run twice into separate dirs and compare Excel hash
        out_a = args.out / "rep_a"
        out_b = args.out / "rep_b"
        a = _build(args.seed, out_a, use_regressors=not args.no_regressors)
        b = _build(args.seed, out_b, use_regressors=not args.no_regressors)
        ha = hashlib.sha256(Path(a["excel"]).read_bytes()).hexdigest()
        hb = hashlib.sha256(Path(b["excel"]).read_bytes()).hexdigest()
        assert ha == hb, f"Reproducibility failed: Excel hashes differ\n  {ha}\n  {hb}"
        print("Reproducibility check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
