"""PyInstaller entry point.

PyInstaller runs the entry script as __main__, which breaks the relative imports
in app/main.py. This launcher imports the app package properly so those relative
imports resolve. Build targets this file; day-to-day dev still uses
`python -m app.main`.
"""
from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
