"""Design system — theme tokens + QSS stylesheet for the entire app.

Single source of truth so the design stays consistent across tabs. Edit colours and
typography here, never inline.
"""
from __future__ import annotations

# Tokens (see plan §GUI visual design)
COLOURS = {
    "bg_page": "#F9FAFB",
    "bg_surface": "#FFFFFF",
    "bg_sidebar": "#1F2937",
    "bg_sidebar_hover": "#374151",
    "border": "#E5E7EB",
    "text_primary": "#111827",
    "text_secondary": "#6B7280",
    "text_on_dark": "#F9FAFB",
    "accent_amber": "#FCD34D",
    "accent_amber_dk": "#F59E0B",
    "positive_green": "#10B981",
    "negative_red": "#DC2626",
    "info_blue": "#3B82F6",
}

SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
FONT_FAMILY = "Inter, 'Segoe UI', system-ui, sans-serif"

QSS = """
QWidget {{
    background-color: {bg_page};
    color: {text_primary};
    font-family: {font_family};
    font-size: 14px;
}}

QLabel#TabTitle {{
    font-size: 28px;
    font-weight: 600;
    color: {text_primary};
    padding-bottom: 8px;
}}

QLabel#H2 {{
    font-size: 20px;
    font-weight: 600;
    color: {text_primary};
}}

QLabel#Caption {{
    font-size: 12px;
    color: {text_secondary};
}}

QLabel#KpiValue {{
    font-size: 28px;
    font-weight: 600;
    color: {text_primary};
}}

QFrame#Card {{
    background-color: {bg_surface};
    border: 1px solid {border};
    border-radius: 8px;
}}

QFrame#Sidebar {{
    background-color: {bg_sidebar};
    border: none;
}}

QPushButton#NavButton {{
    color: {text_on_dark};
    background: transparent;
    border: none;
    padding: 12px 16px;
    text-align: left;
    font-size: 14px;
}}
QPushButton#NavButton:hover {{
    background-color: {bg_sidebar_hover};
}}
QPushButton#NavButton:checked {{
    background-color: {bg_sidebar_hover};
    border-left: 3px solid {accent_amber};
}}

QPushButton#Primary {{
    background-color: {bg_sidebar};
    color: {text_on_dark};
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background-color: {bg_sidebar_hover}; }}
QPushButton#Secondary {{
    background-color: {bg_surface};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 16px;
}}
QPushButton#Secondary:hover {{ background-color: {bg_page}; }}
QPushButton#Destructive {{
    background-color: {negative_red};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}}

QTableView {{
    background-color: {bg_surface};
    alternate-background-color: {bg_page};
    selection-background-color: {accent_amber};
    selection-color: {text_primary};
    gridline-color: {border};
    border: 1px solid {border};
    border-radius: 6px;
}}
QHeaderView::section {{
    background-color: {bg_page};
    color: {text_secondary};
    text-transform: uppercase;
    font-size: 11px;
    font-weight: 600;
    padding: 8px;
    border: none;
    border-bottom: 1px solid {border};
}}

QProgressBar {{
    background-color: {border};
    border-radius: 4px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {accent_amber};
    border-radius: 4px;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {bg_surface};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {accent_amber};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {accent_amber};
}}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 4px;
    border: 1px solid {border}; background: {bg_surface}; }}
QCheckBox::indicator:checked {{ background: {accent_amber}; border: 1px solid {accent_amber}; }}
"""


def stylesheet() -> str:
    """Return the fully-rendered QSS string."""
    return QSS.format(**{**COLOURS, "font_family": FONT_FAMILY})
