"""Models tab — checkbox list + Run, results table ranked by backtest RMSE.

Synchronous in v1 (the panel fits in seconds for typical input sizes). v1.1 will move
the fit to a QThreadPool with per-model progress signals.
"""
from __future__ import annotations

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QCheckBox,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False

import pandas as pd

from ..core.backtest import aggregate_metrics, pooled_residual_sigma, walk_forward
from ..core.blending import blend_forecasts
from ..core.models import default_panel, fit_and_forecast
from ..core.regressors import climatology_future
from ._common import make_tab_scaffold

if PYQT_AVAILABLE:
    class ModelTab(QWidget):
        def __init__(self, state, parent=None) -> None:
            super().__init__(parent)
            self.state = state
            page, v = make_tab_scaffold("Models")
            outer = page.layout()

            top = QHBoxLayout()
            self.run_btn = QPushButton("Run Selected")
            self.run_btn.setObjectName("Primary")
            self.run_btn.clicked.connect(self._run)
            self.run_all_btn = QPushButton("Run All")
            self.run_all_btn.setObjectName("Secondary")
            self.run_all_btn.clicked.connect(self._run_all)
            top.addWidget(self.run_btn)
            top.addWidget(self.run_all_btn)
            top.addStretch(1)
            outer.addLayout(top)

            self._checkboxes: dict[str, QCheckBox] = {}
            check_row = QHBoxLayout()
            for m in default_panel(advanced=self.state.settings.get("advanced", False)):
                cb = QCheckBox(m.name)
                cb.setChecked(True)
                self._checkboxes[m.name] = cb
                check_row.addWidget(cb)
            check_row.addStretch(1)
            outer.addLayout(check_row)

            self.status = QLabel("Load data on the Data tab, then click Run All.")
            outer.addWidget(self.status)

            self.results = QTableWidget(0, 8, self)
            self.results.setHorizontalHeaderLabels([
                "Model", "Backtest RMSE", "MAE", "MAPE", "MASE",
                "Bias", "P20-P80 cov.", "Fit (s)",
            ])
            self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            outer.addWidget(self.results, 1)

            shell = QHBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.addWidget(page)

        def _run_all(self) -> None:
            for cb in self._checkboxes.values():
                cb.setChecked(True)
            self._run()

        def _run(self) -> None:
            if self.state.history is None:
                QMessageBox.warning(self, "No data", "Load data on the Data tab first.")
                return
            self.status.setText("Running backtest…")
            self._do_run()

        def _do_run(self) -> None:
            history = self.state.history.df
            regressors = self.state.regressors
            advanced = self.state.settings.get("advanced", False)
            selected = [m for m in default_panel(advanced=advanced)
                        if self._checkboxes.get(m.name, None) is not None
                        and self._checkboxes[m.name].isChecked()]
            folds = walk_forward(
                history, selected,
                holdout_years=self.state.settings.get("holdout_years", 1),
                regressors=regressors,
            )
            self.state.fold_results = folds
            metrics = aggregate_metrics(folds)
            self._populate_table(metrics)

            # Build forward forecasts for the blended view
            future_regs = climatology_future(regressors, 52) if regressors is not None else None
            forecasts: dict[str, pd.DataFrame] = {}
            for model_tpl in selected:
                model = model_tpl.__class__()
                sigma = pooled_residual_sigma(folds, model.name) if folds else None
                if sigma is not None and len(sigma) == 0:
                    sigma = None
                fc = fit_and_forecast(model, history, horizon=52,
                                      regressors=regressors, future_regressors=future_regs,
                                      residual_sigma=sigma)
                if not fc.failed:
                    forecasts[model.name] = fc.forecast
            self.state.forecasts = forecasts
            self.state.blend = blend_forecasts(folds, forecasts, strategy="inverse_rmse")
            self.status.setText(f"Done. Blend strategy: {self.state.blend.chosen_strategy} "
                                f"(backtest RMSE {self.state.blend.backtest_rmse:.3f}).")

        def _populate_table(self, metrics: pd.DataFrame) -> None:
            self.results.setRowCount(len(metrics))
            for i, (_, row) in enumerate(metrics.iterrows()):
                vals = [
                    row["model"],
                    f"{row['rmse']:.3f}" if pd.notna(row["rmse"]) else "—",
                    f"{row['mae']:.3f}" if pd.notna(row["mae"]) else "—",
                    f"{row['mape']*100:.1f}%" if pd.notna(row["mape"]) else "—",
                    f"{row['mase']:.3f}" if pd.notna(row["mase"]) else "—",
                    f"{row['bias']:+.3f}" if pd.notna(row["bias"]) else "—",
                    f"{row['coverage_p20_p80']*100:.0f}%" if pd.notna(row["coverage_p20_p80"]) else "—",
                    f"{row['fit_seconds']:.2f}",
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    if i == 0 and not row.get("failed", False):
                        item.setBackground(QColor("#D1FAE5"))  # winning row green
                    if row.get("failed", False):
                        item.setForeground(QColor("#6B7280"))
                    self.results.setItem(i, j, item)
            self.results.resizeColumnsToContents()
else:
    class ModelTab:  # pragma: no cover
        def __init__(self, *_, **__): pass
