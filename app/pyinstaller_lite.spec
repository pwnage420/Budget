# PyInstaller spec — lite build (no PyTorch / darts / pymc).
#
# Build from the repo root:  pyinstaller app/pyinstaller_lite.spec --clean --noconfirm
#
# Target: single Windows .exe under 250 MB. Excludes the heavy Advanced-toggle
# models so the GM gets a fast launcher.
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hidden_imports = (
    collect_submodules("statsmodels")
    + collect_submodules("xgboost")
    + collect_submodules("lightgbm")
    + ["openpyxl", "reportlab", "plotly", "kaleido", "scipy.special"]
)
data_files = (
    collect_data_files("plotly")
    + [(os.path.join(ROOT, "app", "data"), "app/data")]
)

a = Analysis(
    [os.path.join(ROOT, "run_app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    excludes=[
        "torch", "darts", "pmdarima", "prophet", "pymc",
        "pytensor", "shap", "tensorflow",
    ],
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="tomato-forecaster-lite",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
