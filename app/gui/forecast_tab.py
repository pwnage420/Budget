"""Forecast tab — interactive Plotly chart of the blended P50 + bands + history.

Includes export buttons (Excel + PDF).
"""
from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False


from ..core.backtest import aggregate_metrics
from ..export.excel_writer import ExportPayload, export_workbook
from ..export.pdf_writer import PdfPayload, export_pdf
from ._common import make_tab_scaffold
from .plotly_view import PlotlyView

if PYQT_AVAILABLE:
    class ForecastTab(QWidget):
        def __init__(self, state, parent=None) -> None:
            super().__init__(parent)
            self.state = state
            page, v = make_tab_scaffold("Forecast")
            outer = page.layout()

            top = QHBoxLayout()
            self.refresh_btn = QPushButton("Refresh chart")
            self.refresh_btn.setObjectName("Primary")
            self.refresh_btn.clicked.connect(self._refresh)
            self.excel_btn = QPushButton("Export Excel…")
            self.excel_btn.setObjectName("Secondary")
            self.excel_btn.clicked.connect(self._export_excel)
            self.pdf_btn = QPushButton("Export PDF…")
            self.pdf_btn.setObjectName("Secondary")
            self.pdf_btn.clicked.connect(self._export_pdf)
            top.addWidget(self.refresh_btn)
            top.addWidget(self.excel_btn)
            top.addWidget(self.pdf_btn)
            top.addStretch(1)
            outer.addLayout(top)

            self.status = QLabel("Run the model panel first (Models tab).")
            outer.addWidget(self.status)

            self.plot = PlotlyView(self)
            outer.addWidget(self.plot, 1)

            shell = QHBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.addWidget(page)

        def _refresh(self) -> None:
            if self.state.blend is None or self.state.blend.forecast.empty:
                QMessageBox.information(self, "No forecast",
                                         "Run the model panel on the Models tab first.")
                return
            fig = self._build_figure()
            self.plot.show_figure(fig)
            self.status.setText(
                f"Blend strategy: {self.state.blend.chosen_strategy} "
                f"({len(self.state.blend.weights)} models) — backtest RMSE "
                f"{self.state.blend.backtest_rmse:.3f}."
            )

        def _build_figure(self):
            import plotly.graph_objects as go
            hist = self.state.history.df
            fc = self.state.blend.forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], mode="lines",
                                      line=dict(color="#9CA3AF", width=1.5), name="History"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["p80"], mode="lines",
                                      line=dict(color="rgba(252,211,77,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["p20"], mode="lines",
                                      fill="tonexty", fillcolor="rgba(252,211,77,0.35)",
                                      line=dict(color="rgba(252,211,77,0)"), name="P20–P80"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["p70"], mode="lines",
                                      line=dict(color="rgba(245,158,11,0)"), showlegend=False))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["p30"], mode="lines",
                                      fill="tonexty", fillcolor="rgba(245,158,11,0.45)",
                                      line=dict(color="rgba(245,158,11,0)"), name="P30–P70"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["mean"], mode="lines",
                                      line=dict(color="#111827", width=2.5), name="P50"))
            fig.update_layout(
                template="simple_white",
                paper_bgcolor="#F9FAFB", plot_bgcolor="#FFFFFF",
                font=dict(family="Inter, Segoe UI, sans-serif", color="#111827"),
                margin=dict(l=40, r=20, t=20, b=40),
                yaxis_title="$/kg",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            return fig

        def _export_excel(self) -> None:
            if self.state.blend is None:
                QMessageBox.information(self, "No forecast", "Run the model panel first.")
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Excel workbook", "tomato_forecast.xlsx", "Excel (*.xlsx)")
            if not path:
                return
            payload = ExportPayload(
                history=self.state.history.df,
                forecast=self.state.blend.forecast,
                metrics=aggregate_metrics(self.state.fold_results),
                blend_weights=self.state.blend.weights,
                blend_strategy=self.state.blend.chosen_strategy,
                blend_backtest_rmse=self.state.blend.backtest_rmse,
                actuals=self.state.actuals,
            )
            export_workbook(payload, path)
            QMessageBox.information(self, "Exported", f"Saved to {path}")

        def _export_pdf(self) -> None:
            if self.state.blend is None:
                QMessageBox.information(self, "No forecast", "Run the model panel first.")
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "Save PDF report", "tomato_forecast.pdf", "PDF (*.pdf)")
            if not path:
                return
            payload = PdfPayload(
                history=self.state.history.df,
                forecast=self.state.blend.forecast,
                actuals=self.state.actuals,
                metrics=aggregate_metrics(self.state.fold_results),
                blend_weights=self.state.blend.weights,
                blend_strategy=self.state.blend.chosen_strategy,
                blend_backtest_rmse=self.state.blend.backtest_rmse,
            )
            export_pdf(payload, path)
            QMessageBox.information(self, "Exported", f"Saved to {path}")
else:
    class ForecastTab:  # pragma: no cover
        def __init__(self, *_, **__): pass
