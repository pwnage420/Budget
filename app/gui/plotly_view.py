"""QWebEngineView wrapper that renders Plotly figures via local file:// URLs.

Security: never interpolates user content into raw HTML. Plotly serialises figures
to JSON; we embed plotly.js inline so the app remains fully offline (plan §Security
item 4).

Graceful degradation: QtWebEngine (full Chromium) needs more system libraries than
QtWidgets. If QtWidgets is present but QtWebEngine isn't, PlotlyView falls back to a
QLabel-based placeholder (still a real QWidget so layouts don't break). If PyQt6
isn't present at all (headless/CI), PlotlyView is a no-op stub.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

_WEBENGINE = False
_QTWIDGETS = False
try:
    from PyQt6.QtWidgets import QLabel  # noqa: F401
    _QTWIDGETS = True
except ImportError:
    pass
try:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    _WEBENGINE = True
except ImportError:
    pass

# Back-compat flag used by other modules
PYQT_AVAILABLE = _WEBENGINE


def figure_to_html(fig, embed_js: bool = True) -> str:
    """Convert a Plotly figure to a self-contained HTML page."""
    import plotly.io as pio
    return pio.to_html(
        fig,
        include_plotlyjs=("inline" if embed_js else "cdn"),
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )


if _WEBENGINE:
    class PlotlyView(QWebEngineView):
        """A QWebEngineView that loads a Plotly figure from a temp HTML file."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self._tmp_path: Path | None = None

        def show_figure(self, fig) -> None:
            html = figure_to_html(fig, embed_js=True)
            if self._tmp_path is None:
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".html", delete=False, encoding="utf-8")
                self._tmp_path = Path(tmp.name)
                tmp.close()
            self._tmp_path.write_text(html, encoding="utf-8")
            self.load(QUrl.fromLocalFile(str(self._tmp_path)))

        def clear_figure(self) -> None:
            self.setHtml("<html><body style='background:#F9FAFB'></body></html>")

elif _QTWIDGETS:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel

    class PlotlyView(QLabel):  # type: ignore[no-redef]
        """Fallback widget when QtWebEngine isn't available (e.g. missing system libs).

        Still a real QWidget so layouts work; shows a plain-English message instead
        of an interactive chart. Use the Excel/PDF export to view the forecast chart.
        """

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setWordWrap(True)
            self.setText(
                "Interactive chart unavailable on this machine\n"
                "(QtWebEngine component not installed).\n\n"
                "Use Export PDF or Export Excel to view the forecast chart."
            )
            self.setStyleSheet("color:#6B7280; font-size:14px;")

        def show_figure(self, fig) -> None:  # noqa: ARG002
            pass

        def clear_figure(self) -> None:
            pass

else:
    class PlotlyView:  # type: ignore[no-redef]
        """Stub used when PyQt6 isn't installed at all (headless / CI)."""
        def __init__(self, *_, **__): pass
        def show_figure(self, fig): pass
        def clear_figure(self): pass
