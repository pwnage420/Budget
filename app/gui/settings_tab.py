"""Settings tab — FY config + band z-scores + Advanced toggle + per-model timeout."""
from __future__ import annotations

try:
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False

from ._common import Card, make_tab_scaffold

if PYQT_AVAILABLE:
    class SettingsTab(QWidget):
        def __init__(self, state, parent=None) -> None:
            super().__init__(parent)
            self.state = state
            page, v = make_tab_scaffold("Settings")
            outer = page.layout()

            card = Card(self)
            form = QFormLayout()
            self.fy_month = QComboBox()
            for m in ["January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]:
                self.fy_month.addItem(m)
            self.fy_month.setCurrentIndex(state.settings["fy_start_month"] - 1)
            self.holdout = QSpinBox()
            self.holdout.setRange(1, 5)
            self.holdout.setValue(state.settings["holdout_years"])
            self.z_p20 = QDoubleSpinBox()
            self.z_p20.setRange(0.1, 3.0)
            self.z_p20.setSingleStep(0.05)
            self.z_p20.setValue(state.settings["z_p20_p80"])
            self.z_p30 = QDoubleSpinBox()
            self.z_p30.setRange(0.1, 3.0)
            self.z_p30.setSingleStep(0.05)
            self.z_p30.setValue(state.settings["z_p30_p70"])
            self.advanced = QCheckBox(
                "Enable Advanced models (Prophet, Auto-SARIMA, N-BEATS, TFT, BSTS — slow)")
            self.advanced.setChecked(state.settings["advanced"])
            self.timeout = QSpinBox()
            self.timeout.setRange(5, 600)
            self.timeout.setValue(state.settings["model_timeout_s"])

            form.addRow("Financial Year start month:", self.fy_month)
            form.addRow("Backtest holdout years:", self.holdout)
            form.addRow("Band z-score (P20 / P80):", self.z_p20)
            form.addRow("Band z-score (P30 / P70):", self.z_p30)
            form.addRow("Model timeout (seconds):", self.timeout)
            form.addRow(self.advanced)
            card.layout().addLayout(form)

            self.save_btn = QPushButton("Save")
            self.save_btn.setObjectName("Primary")
            self.save_btn.clicked.connect(self._save)
            card.add(self.save_btn)
            outer.addWidget(card)
            outer.addStretch(1)

            shell = QHBoxLayout(self)
            shell.setContentsMargins(0, 0, 0, 0)
            shell.addWidget(page)

        def _save(self) -> None:
            self.state.settings.update({
                "fy_start_month": self.fy_month.currentIndex() + 1,
                "holdout_years": self.holdout.value(),
                "z_p20_p80": self.z_p20.value(),
                "z_p30_p70": self.z_p30.value(),
                "advanced": self.advanced.isChecked(),
                "model_timeout_s": self.timeout.value(),
            })
else:
    class SettingsTab:  # pragma: no cover
        def __init__(self, *_, **__): pass
