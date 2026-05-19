"""QWebEngineView wrapper that renders Plotly figures via local file:// URLs.

Security: never interpolates user content into raw HTML. Plotly serialises figures
to JSON; the JSON is embedded via `plotly.io.to_html(..., include_plotlyjs="cdn")`
in dev mode, but in production we set `include_plotlyjs=True` (inline) so the app
remains fully offline. See plan §Security item 4.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

try:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover — GUI optional for headless runs
    PYQT_AVAILABLE = False


def figure_to_html(fig, embed_js: bool = True) -> str:
    """Convert a Plotly figure to a self-contained HTML page."""
    import plotly.io as pio
    return pio.to_html(
        fig,
        include_plotlyjs=("inline" if embed_js else "cdn"),
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )


if PYQT_AVAILABLE:
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
else:
    class PlotlyView:  # type: ignore[no-redef]
        """Stub used when PyQt6 isn't installed (headless / CI)."""
        def __init__(self, *_, **__): pass
        def show_figure(self, fig): pass
        def clear_figure(self): pass
