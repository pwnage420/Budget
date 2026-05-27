"""Regressors tab — upload weekly external series, show correlations with price."""
from __future__ import annotations

try:
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

from ..core.data_loader import DataLoadError
from ..core.regressors import correlation_with_price, load_regressors
from ..core.sample_data import generate_regressors
from ._common import make_tab_scaffold

if PYQT_AVAILABLE:
    class RegressorsTab(QWidget):
        def __init__(self, state, parent=None) -> None:
            super().__init__(parent)
            self.state = state
            page, v = make_tab_scaffold("Regressors")
            outer = page.layout()

            top = QHBoxLayout()
            self.load_btn = QPushButton("Load regressors CSV…")
            self.load_btn.setObjectName("Primary")
            self.load_btn.clicked.connect(self._load_file)
            self.sample_btn = QPushButton("Use sample regressors")
            self.sample_btn.setObjectName("Secondary")
            self.sample_btn.clicked.connect(self._load_sample)
            top.addWidget(self.load_btn)
            top.addWidget(self.sample_btn)
            top.addStretch(1)
            outer.addLayout(top)

            self.status = QLabel("No regressors loaded. Optional — improves model accuracy when present.")
            self.status.setWordWrap(True)
            outer.addWidget(self.status)

            self.corr_table = QTableWidget(0, 2, self)
            self.corr_table.setHorizontalHeaderLabels(["Regressor", "Correlation with price"])
            outer.addWidget(self.corr_table, 1)

            shell = QHBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.addWidget(page)

        def _load_file(self) -> None:
            path, _ = QFileDialog.getOpenFileName(self, "Open regressors file",
                                                   filter="Data files (*.csv *.xlsx *.xls)")
            if not path:
                return
            try:
                regs = load_regressors(path)
            except DataLoadError as exc:
                QMessageBox.warning(self, "Could not load", str(exc))
                return
            self._apply(regs)

        def _load_sample(self) -> None:
            regs = generate_regressors()
            self._apply(regs)

        def _apply(self, regs) -> None:
            self.state.regressors = regs
            self.status.setText(
                f"Loaded {len(regs)} weekly regressor rows across "
                f"{regs['year'].nunique()} years and {len(regs.columns) - 3} regressor columns."
            )
            history = self.state.history.df if self.state.history is not None else None
            if history is None:
                self.corr_table.setRowCount(0)
                return
            corr = correlation_with_price(history, regs)
            self.corr_table.setRowCount(len(corr))
            for i, (name, val) in enumerate(corr.items()):
                self.corr_table.setItem(i, 0, QTableWidgetItem(str(name)))
                self.corr_table.setItem(i, 1, QTableWidgetItem(f"{val:+.3f}"))
            self.corr_table.resizeColumnsToContents()
else:
    class RegressorsTab:  # pragma: no cover
        def __init__(self, *_, **__): pass
