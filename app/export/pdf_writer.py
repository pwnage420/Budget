"""PDF export — 4-page board pack: cover, forecast chart, variance commentary, methodology footnote.

Uses reportlab. The Plotly figure is rendered to PNG via kaleido for the chart page;
if kaleido isn't available we render a matplotlib equivalent so the PDF still ships.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CHARCOAL = colors.HexColor("#1F2937")
AMBER = colors.HexColor("#FCD34D")
LIGHT_GREY = colors.HexColor("#F3F4F6")
TEXT_GREY = colors.HexColor("#6B7280")
PRIMARY = colors.HexColor("#111827")


@dataclass
class PdfPayload:
    history: pd.DataFrame
    forecast: pd.DataFrame
    actuals: pd.DataFrame | None
    metrics: pd.DataFrame
    blend_weights: dict[str, float]
    blend_strategy: str
    blend_backtest_rmse: float
    fy_label: str = "FY 2026"


def export_pdf(payload: PdfPayload, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Goodness Grown — Tomato Price Forecast {payload.fy_label}",
    )
    story = []
    story += _cover_page(payload)
    story.append(PageBreak())
    story += _forecast_page(payload)
    story.append(PageBreak())
    story += _variance_page(payload)
    story.append(PageBreak())
    story += _methodology_page(payload)
    doc.build(story)
    return path


def _styles():
    base = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold",
                           fontSize=28, leading=34, textColor=PRIMARY, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold",
                        fontSize=16, leading=20, textColor=PRIMARY, spaceAfter=8)
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica",
                          fontSize=11, leading=16, textColor=PRIMARY, spaceAfter=8)
    small = ParagraphStyle("small", parent=base["BodyText"], fontName="Helvetica",
                           fontSize=9, leading=12, textColor=TEXT_GREY)
    return title, h2, body, small


def _cover_page(payload: PdfPayload) -> list:
    title, h2, body, small = _styles()
    out = [
        Spacer(1, 4 * cm),
        Paragraph("Goodness Grown", title),
        Paragraph(f"Tomato Price Forecast — {payload.fy_label}", h2),
        Spacer(1, 0.6 * cm),
        Paragraph(
            "Weekly retail $/kg forecast with calibrated probability bands, produced "
            "by the desktop forecasting app that replaced the v4 Excel budget model.",
            body),
        Spacer(1, 0.4 * cm),
    ]
    kpi_rows = [
        ["Strategy", payload.blend_strategy],
        ["Backtest RMSE ($/kg)", f"{payload.blend_backtest_rmse:.3f}"
            if payload.blend_backtest_rmse == payload.blend_backtest_rmse else "n/a"],
        ["Models in blend", ", ".join(payload.blend_weights.keys()) or "n/a"],
        ["Horizon", f"{len(payload.forecast)} weeks"],
        ["Generated", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")],
    ]
    t = Table(kpi_rows, colWidths=[6 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT_GREY),
        ("TEXTCOLOR", (1, 0), (1, -1), PRIMARY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    out.append(t)
    out += [
        Spacer(1, 3 * cm),
        Paragraph(
            "Methodology summary and full glossary are on page 4. "
            "Cross-reference the Excel export for week-by-week numbers and the FY "
            "Tracking sheet.", small),
    ]
    return out


def _forecast_page(payload: PdfPayload) -> list:
    title, h2, body, small = _styles()
    out = [Paragraph("Forecast — next 52 weeks", h2)]
    img = _render_forecast_png(payload)
    if img is not None:
        out.append(Image(img, width=17 * cm, height=10 * cm))
    out.append(Spacer(1, 0.4 * cm))
    out.append(Paragraph(
        "Black line is the P50 (median) forecast. The amber band is P20–P80; the "
        "darker band inside is P30–P70. Grey points are historical actuals. The red "
        "line, if shown, is the current-year budget overlay.", small))
    return out


def _variance_page(payload: PdfPayload) -> list:
    title, h2, body, small = _styles()
    out = [Paragraph("Variance commentary", h2)]
    if payload.actuals is None or payload.actuals.empty:
        out.append(Paragraph("No FY actuals have been entered yet. This section will populate "
                             "automatically once weekly actuals are recorded in the FY Tracking "
                             "tab.", body))
        return out
    a = payload.actuals.copy()
    a["date"] = pd.to_datetime(a["date"])
    fc = payload.forecast.copy()
    fc["date"] = pd.to_datetime(fc["date"])
    merged = a.merge(fc, on="date", how="inner")
    if merged.empty:
        out.append(Paragraph("FY actuals don't overlap with the current forecast window.", body))
        return out
    merged["variance_$"] = merged["actual_price"] - merged["mean"]
    merged["variance_pct"] = merged["variance_$"] / merged["mean"] * 100
    avg_actual = float(merged["actual_price"].mean())
    avg_p50 = float(merged["mean"].mean())
    coverage = float(((merged["actual_price"] >= merged["p20"]) &
                      (merged["actual_price"] <= merged["p80"])).mean())
    rows = [
        ["Weeks entered", str(len(merged))],
        ["FY-to-date avg actual", f"${avg_actual:.2f}/kg"],
        ["FY-to-date avg P50", f"${avg_p50:.2f}/kg"],
        ["Variance (avg)", f"${(avg_actual - avg_p50):+.2f}/kg "
                          f"({(avg_actual - avg_p50) / avg_p50 * 100:+.1f}%)"],
        ["Band coverage (P20–P80)", f"{coverage:.0%} (target ≈60%)"],
    ]
    t = Table(rows, colWidths=[6 * cm, 9 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT_GREY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
    ]))
    out.append(t)
    return out


def _methodology_page(payload: PdfPayload) -> list:
    title, h2, body, small = _styles()
    return [
        Paragraph("Methodology", h2),
        Paragraph(
            "Weekly retail tomato prices were fitted with a panel of models including "
            "statistical baselines (Naive, Seasonal + Trend, Holt-Winters, SARIMA), "
            "machine-learning models (XGBoost, LightGBM, Random Forest), and optionally "
            "Prophet, Bayesian Structural Time Series, N-BEATS and TFT when the "
            "Advanced toggle is enabled. Each model was backtested using walk-forward "
            "(expanding-window) cross-validation on the most recent holdout year(s) "
            "and ranked by out-of-sample RMSE.", body),
        Paragraph(
            "The top 3 models were blended by inverse-RMSE weighting. If the blended "
            "model did not beat the best single model on backtest, the best single "
            "model was used (spec guardrail). Probability bands come from the residual "
            "standard deviation pooled across folds and nearby horizon steps, then "
            "rescaled so the historical P20–P80 band covers about 60% of holdout actuals.",
            body),
        Paragraph(
            "Citations: Hyndman & Athanasopoulos (FPP3, 2021); Taylor & Letham "
            "(Prophet, 2018); Chen & Guestrin (XGBoost, 2016); Ke et al. (LightGBM, "
            "2017); Bates & Granger (combination, 1969); Wolpert (stacking, 1992); "
            "Gneiting & Raftery (proper scoring, 2007). See the app README for full "
            "list and limitations.", small),
        Paragraph("Glossary: P50 = median forecast. P20/P80 = 20th and 80th percentiles. "
                  "RMSE = root mean squared error (lower is better). FY = AU financial year (Jul–Jun). "
                  "Walk-forward CV = train on past, test on the next holdout year, repeat.", small),
    ]


def _render_forecast_png(payload: PdfPayload) -> io.BytesIO | None:
    """Render the forecast chart to PNG. Tries kaleido (Plotly); falls back to matplotlib."""
    buf = io.BytesIO()
    try:
        fig = _plotly_forecast_figure(payload)
        fig.write_image(buf, format="png", width=1100, height=600, scale=2)
        buf.seek(0)
        return buf
    except Exception:
        pass
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
        hist = payload.history.copy()
        ax.plot(pd.to_datetime(hist["date"]), hist["price"], color="#9CA3AF", lw=1,
                label="History")
        fc = payload.forecast.copy()
        ax.fill_between(pd.to_datetime(fc["date"]), fc["p20"], fc["p80"],
                        color="#FCD34D", alpha=0.45, label="P20–P80")
        ax.fill_between(pd.to_datetime(fc["date"]), fc["p30"], fc["p70"],
                        color="#F59E0B", alpha=0.45, label="P30–P70")
        ax.plot(pd.to_datetime(fc["date"]), fc["mean"], color="#111827", lw=2,
                label="P50")
        if payload.actuals is not None and not payload.actuals.empty:
            ax.scatter(pd.to_datetime(payload.actuals["date"]),
                       payload.actuals["actual_price"],
                       color="#DC2626", s=18, label="Actuals", zorder=5)
        ax.set_ylabel("$/kg")
        ax.set_title("Forecast — next 52 weeks")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        return None


def _plotly_forecast_figure(payload: PdfPayload):
    import plotly.graph_objects as go
    fig = go.Figure()
    hist = payload.history.copy()
    fc = payload.forecast.copy()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(hist["date"]), y=hist["price"], mode="lines",
        name="History", line=dict(color="#9CA3AF", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(fc["date"]), y=fc["p80"], mode="lines",
        line=dict(color="rgba(252,211,77,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(fc["date"]), y=fc["p20"], mode="lines",
        fill="tonexty", fillcolor="rgba(252,211,77,0.35)",
        line=dict(color="rgba(252,211,77,0)"), name="P20–P80",
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(fc["date"]), y=fc["p70"], mode="lines",
        line=dict(color="rgba(245,158,11,0)"), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(fc["date"]), y=fc["p30"], mode="lines",
        fill="tonexty", fillcolor="rgba(245,158,11,0.45)",
        line=dict(color="rgba(245,158,11,0)"), name="P30–P70",
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(fc["date"]), y=fc["mean"], mode="lines",
        name="P50 forecast", line=dict(color="#111827", width=2.5),
    ))
    if payload.actuals is not None and not payload.actuals.empty:
        a = payload.actuals.copy()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(a["date"]), y=a["actual_price"], mode="markers",
            marker=dict(color="#DC2626", size=8), name="Actuals",
        ))
    fig.update_layout(
        template="simple_white",
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="#F9FAFB", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#111827"),
        title="Forecast — next 52 weeks",
        yaxis_title="$/kg",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
