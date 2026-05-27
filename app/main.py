"""GUI entry point.

Run with:  python -m app.main
"""
from __future__ import annotations

import os
import sys

from .core.seeding import seed_all


def _configure_frozen_runtime() -> None:
    """When running as a PyInstaller .exe, force PyMC's PyTensor C backend off.

    PyTensor JIT-compiles C at runtime, which has no compiler in a frozen app.
    The Python fallback is slower but works. No-op when not frozen.
    """
    if getattr(sys, "frozen", False):
        os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")


def main() -> int:
    _configure_frozen_runtime()
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
