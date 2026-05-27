"""FY Tracking tab — editable actuals + variance computed live + KPI tiles."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False

import pandas as pd

from ..core.audit import app_home
from ._common import KpiTile, make_tab_scaffold


def _actuals_path(fy_label: str) -> Path:
    d = app_home() / "actuals"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{fy_label}.json"


if PYQT_AVAILABLE:
    class TrackingTab(QWidget):
        def __init__(self, state, parent=None) -> None:
            super().__init__(parent)
            self.state = state
            page, v = make_tab_scaffold("FY Tracking")
            outer = page.layout()

            kpi_row = QGridLayout()
            self.kpi_weeks = KpiTile("Weeks entered", "0")
            self.kpi_avg_actual = KpiTile("FYTD avg actual", "—")
            self.kpi_avg_p50 = KpiTile("FYTD avg P50", "—")
            self.kpi_variance = KpiTile("Variance", "—")
            self.kpi_coverage = KpiTile("Coverage (P20-P80)", "—")
            for i, t in enumerate([self.kpi_weeks, self.kpi_avg_actual, self.kpi_avg_p50,
                                    self.kpi_variance, self.kpi_coverage]):
                kpi_row.addWidget(t, 0, i)
            outer.addLayout(kpi_row)

            top = QHBoxLayout()
            self.save_btn = QPushButton("Save actuals")
            self.save_btn.setObjectName("Primary")
            self.save_btn.clicked.connect(self._save)
            self.reforecast_btn = QPushButton("Re-forecast")
            self.reforecast_btn.setObjectName("Secondary")
            self.reforecast_btn.setEnabled(False)
            top.addWidget(self.save_btn)
            top.addWidget(self.reforecast_btn)
            top.addStretch(1)
            outer.addLayout(top)

            self.table = QTableWidget(0, 7, self)
            self.table.setHorizontalHeaderLabels([
                "Week start", "Year", "Week", "P50", "Actual", "Variance ($)", "Position",
            ])
            self.table.itemChanged.connect(self._on_edit)
            outer.addWidget(self.table, 1)

            shell = QHBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.addWidget(page)

            self.refresh()

        def refresh(self) -> None:
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            if self.state.blend is None or self.state.blend.forecast.empty:
                self.table.blockSignals(False)
                return
            fc = self.state.blend.forecast.copy()
            actuals = self._load_actuals()
            actuals_map = {pd.Timestamp(k).date(): v for k, v in actuals.items()}
            self.table.setRowCount(len(fc))
            for i, (_, row) in enumerate(fc.iterrows()):
                d = pd.Timestamp(row["date"])
                actual = actuals_map.get(d.date())
                items = [
                    d.strftime("%Y-%m-%d"),
                    str(d.isocalendar().year),
                    str(d.isocalendar().week),
                    f"{row['mean']:.2f}",
                    "" if actual is None else f"{actual:.2f}",
                    "" if actual is None else f"{actual - row['mean']:+.2f}",
                    "" if actual is None else self._position(actual, row),
                ]
                for j, txt in enumerate(items):
                    it = QTableWidgetItem(txt)
                    if j != 4:
                        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(i, j, it)
            self.table.resizeColumnsToContents()
            self.table.blockSignals(False)
            self._update_kpis()

        @staticmethod
        def _position(actual: float, row: pd.Series) -> str:
            if actual < row["p20"]:
                return "Below"
            if actual > row["p80"]:
                return "Above"
            return "Within"

        def _on_edit(self, item) -> None:
            if item.column() != 4:
                return
            try:
                value = float(item.text()) if item.text() else None
            except ValueError:
                item.setText("")
                return
            week_start = self.table.item(item.row(), 0).text()
            actuals = self._load_actuals()
            if value is None:
                actuals.pop(week_start, None)
            else:
                actuals[week_start] = value
            self._store_actuals(actuals)
            self.refresh()

        def _load_actuals(self) -> dict[str, float]:
            p = _actuals_path("current")
            if not p.exists():
                return {}
            try:
                return {str(k): float(v) for k, v in json.loads(p.read_text()).items()}
            except Exception:
                return {}

        def _store_actuals(self, actuals: dict[str, float]) -> None:
            p = _actuals_path("current")
            p.write_text(json.dumps(actuals, indent=2))
            # also surface to shared state for exports
            rows = [{"date": pd.Timestamp(k), "actual_price": float(v)}
                    for k, v in actuals.items()]
            self.state.actuals = pd.DataFrame(rows)

        def _save(self) -> None:
            QMessageBox.information(self, "Saved",
                                     f"Actuals saved to {_actuals_path('current')}")

        def _update_kpis(self) -> None:
            actuals = self.state.actuals
            if actuals is None or actuals.empty:
                self.kpi_weeks.set_value("0")
                return
            self.kpi_weeks.set_value(str(len(actuals)))
            avg_actual = float(actuals["actual_price"].mean())
            self.kpi_avg_actual.set_value(f"${avg_actual:.2f}")
            if self.state.blend is None:
                return
            fc = self.state.blend.forecast.copy()
            merged = actuals.merge(fc, on="date", how="left")
            if merged.empty:
                return
            avg_p50 = float(merged["mean"].mean())
            self.kpi_avg_p50.set_value(f"${avg_p50:.2f}")
            self.kpi_variance.set_value(f"{avg_actual - avg_p50:+.2f}",
                                         f"{(avg_actual - avg_p50) / avg_p50 * 100:+.1f}%")
            inside = ((merged["actual_price"] >= merged["p20"]) &
                      (merged["actual_price"] <= merged["p80"])).mean()
            self.kpi_coverage.set_value(f"{inside*100:.0f}%", "target ≈60%")
else:
    class TrackingTab:  # pragma: no cover
        def __init__(self, *_, **__): pass
