"""Shared widgets used across multiple tabs (cards, KPI tiles, tab scaffolds)."""
from __future__ import annotations

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False


if PYQT_AVAILABLE:
    class Card(QFrame):
        """A card container with the standard surface styling + padding."""

        def __init__(self, parent: QWidget | None = None, padding: int = 16) -> None:
            super().__init__(parent)
            self.setObjectName("Card")
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(padding, padding, padding, padding)
            self._layout.setSpacing(12)

        def add(self, w: QWidget) -> None:
            self._layout.addWidget(w)

        def layout(self):
            return self._layout


    class KpiTile(QFrame):
        def __init__(self, label: str, value: str = "—", caption: str = "",
                     parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setObjectName("Card")
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            v = QVBoxLayout(self)
            v.setContentsMargins(16, 16, 16, 16)
            v.setSpacing(4)
            self.label = QLabel(label)
            self.label.setObjectName("Caption")
            self.value = QLabel(value)
            self.value.setObjectName("KpiValue")
            self.caption = QLabel(caption)
            self.caption.setObjectName("Caption")
            v.addWidget(self.label)
            v.addWidget(self.value)
            v.addWidget(self.caption)

        def set_value(self, value: str, caption: str = "") -> None:
            self.value.setText(value)
            self.caption.setText(caption)


    def make_tab_scaffold(title: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 24, 24, 24)
        v.setSpacing(16)
        heading = QLabel(title)
        heading.setObjectName("TabTitle")
        v.addWidget(heading)
        return page, v
else:
    class Card:  # pragma: no cover
        def __init__(self, *_, **__): pass

    class KpiTile:  # pragma: no cover
        def __init__(self, *_, **__): pass
        def set_value(self, *_, **__): pass

    def make_tab_scaffold(*_, **__):  # pragma: no cover
        return None, None
