"""Main window — shell with dark sidebar + content area + six tabs.

Lazy-imports each tab so a missing optional dep doesn't take the whole GUI down.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False

from .style import stylesheet


@dataclass
class AppState:
    """Shared mutable state across tabs. Owned by MainWindow; tabs hold a reference."""
    history: object | None = None        # LoadedSeries
    regressors: object | None = None     # pd.DataFrame
    fold_results: list = field(default_factory=list)
    forecasts: dict = field(default_factory=dict)
    blend: object | None = None
    actuals: object | None = None
    settings: dict = field(default_factory=lambda: {
        "fy_start_month": 7,
        "holdout_years": 1,
        "z_p20_p80": 0.84,
        "z_p30_p70": 0.52,
        "advanced": False,
        "model_timeout_s": 60,
    })


if PYQT_AVAILABLE:
    class MainWindow(QMainWindow):
        TAB_NAMES = ("Data", "Regressors", "Models", "Forecast", "FY Tracking", "Settings")

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Goodness Grown — Tomato Price Forecaster")
            self.resize(1280, 800)
            self.state = AppState()

            central = QWidget()
            self.setCentralWidget(central)
            outer = QHBoxLayout(central)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)

            self.sidebar = self._build_sidebar()
            self.stack = QStackedWidget()
            outer.addWidget(self.sidebar)
            outer.addWidget(self.stack, 1)

            self._tabs = []
            self._register_tabs()
            self._nav_buttons[0].setChecked(True)
            self.stack.setCurrentIndex(0)

            self.setStyleSheet(stylesheet())

        def _build_sidebar(self) -> QFrame:
            sidebar = QFrame()
            sidebar.setObjectName("Sidebar")
            sidebar.setFixedWidth(240)
            v = QVBoxLayout(sidebar)
            v.setContentsMargins(16, 24, 16, 24)
            v.setSpacing(4)
            title = QLabel("Goodness Grown")
            title.setStyleSheet("color: #F9FAFB; font-size: 16px; font-weight: 600;")
            sub = QLabel("Tomato Price Forecaster")
            sub.setStyleSheet("color: #9CA3AF; font-size: 12px;")
            v.addWidget(title)
            v.addWidget(sub)
            v.addSpacing(16)
            self._nav_buttons = []
            for i, name in enumerate(self.TAB_NAMES):
                btn = QPushButton(name)
                btn.setObjectName("NavButton")
                btn.setCheckable(True)
                btn.clicked.connect(lambda _, idx=i: self._on_nav(idx))
                v.addWidget(btn)
                self._nav_buttons.append(btn)
            v.addStretch(1)
            version = QLabel("v0.1.0")
            version.setStyleSheet("color: #6B7280; font-size: 11px;")
            v.addWidget(version)
            return sidebar

        def _on_nav(self, idx: int) -> None:
            for j, b in enumerate(self._nav_buttons):
                b.setChecked(j == idx)
            self.stack.setCurrentIndex(idx)

        def _register_tabs(self) -> None:
            from .data_tab import DataTab
            from .forecast_tab import ForecastTab
            from .model_tab import ModelTab
            from .regressors_tab import RegressorsTab
            from .settings_tab import SettingsTab
            from .tracking_tab import TrackingTab
            for cls in (DataTab, RegressorsTab, ModelTab, ForecastTab, TrackingTab, SettingsTab):
                tab = cls(self.state, parent=self)
                self.stack.addWidget(tab)
                self._tabs.append(tab)


    def launch() -> int:
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        # Bundled Inter would load here via QFontDatabase; falls back to Segoe UI / system
        win = MainWindow()
        win.show()
        return app.exec()
else:
    class MainWindow:  # pragma: no cover
        pass

    def launch() -> int:  # pragma: no cover
        raise RuntimeError("PyQt6 is not installed. Use 'pip install -r app/requirements.txt'.")
