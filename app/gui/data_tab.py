"""Data tab — file load + validation summary + preview table."""
from __future__ import annotations

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QFileDialog,
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

from ..core.data_loader import (
    DataLoadError,
    LoadedSeries,
    load_series,
    load_series_from_dataframe,
)
from ..core.sample_data import generate_price_series
from ._common import Card, make_tab_scaffold

if PYQT_AVAILABLE:
    class DataTab(QWidget):
        def __init__(self, state, parent=None) -> None:
            super().__init__(parent)
            self.state = state
            page, v = make_tab_scaffold("Data")
            v.parentWidget().setParent(self)  # noqa: ignore
            outer = page.layout()

            # Top row: load buttons + validation summary
            top = QHBoxLayout()
            self.load_btn = QPushButton("Load CSV / Excel…")
            self.load_btn.setObjectName("Primary")
            self.load_btn.clicked.connect(self._load_file)
            self.sample_btn = QPushButton("Use sample data")
            self.sample_btn.setObjectName("Secondary")
            self.sample_btn.clicked.connect(self._load_sample)
            top.addWidget(self.load_btn)
            top.addWidget(self.sample_btn)
            top.addStretch(1)
            outer.addLayout(top)

            self.summary_card = Card(self)
            self.summary_label = QLabel("No data loaded — click 'Use sample data' to begin.")
            self.summary_label.setWordWrap(True)
            self.summary_card.add(self.summary_label)
            outer.addWidget(self.summary_card)

            self.preview = QTableWidget(0, 0, self)
            self.preview.setAlternatingRowColors(True)
            self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            outer.addWidget(self.preview, 1)

            # Place the whole page as our layout
            shell = QHBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.addWidget(page)

        # ------------------------------------------------------------------
        def _load_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open price file",
                filter="Data files (*.csv *.xlsx *.xls);;All files (*)",
            )
            if not path:
                return
            try:
                series = load_series(path)
            except DataLoadError as exc:
                QMessageBox.warning(self, "Could not load data", str(exc))
                return
            self._apply(series)

        def _load_sample(self) -> None:
            df = generate_price_series()
            try:
                series = load_series_from_dataframe(df)
            except DataLoadError as exc:
                QMessageBox.warning(self, "Sample data error", str(exc))
                return
            self._apply(series)

        def _apply(self, series: LoadedSeries) -> None:
            self.state.history = series
            self.summary_label.setText(series.report.summary())
            self._populate_preview(series)

        def _populate_preview(self, series: LoadedSeries) -> None:
            df = series.df.head(50)
            self.preview.setRowCount(len(df))
            self.preview.setColumnCount(len(df.columns))
            self.preview.setHorizontalHeaderLabels([str(c) for c in df.columns])
            outliers = {(y, w) for y, w, _ in series.report.outliers}
            for i, (_, row) in enumerate(df.iterrows()):
                for j, col in enumerate(df.columns):
                    val = row[col]
                    if hasattr(val, "strftime"):
                        val = val.strftime("%Y-%m-%d")
                    item = QTableWidgetItem(str(val))
                    if (int(row["year"]), int(row["week"])) in outliers:
                        item.setBackground(Qt.GlobalColor.yellow)
                    self.preview.setItem(i, j, item)
            self.preview.resizeColumnsToContents()
else:
    class DataTab:  # pragma: no cover
        def __init__(self, *_, **__): pass
