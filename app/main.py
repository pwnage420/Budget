"""GUI entry point.

Run with:  python -m app.main
"""
from __future__ import annotations

import sys

from .core.seeding import seed_all


def main() -> int:
    seed_all(42)
    try:
        from .gui.main_window import launch
    except ImportError as exc:
        sys.stderr.write(
            "PyQt6 is not installed. Run 'pip install -r app/requirements.txt' first, "
            f"or use the headless demo with 'python -m app.demo'. ({exc})\n"
        )
        return 1
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
