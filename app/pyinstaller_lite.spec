# PyInstaller spec — lite build (no PyTorch / darts / pymc).
#
# Build with:  pyinstaller app/pyinstaller_lite.spec
#
# Target: single Windows .exe under 250 MB. Excludes the heavy Advanced toggle
# models so the GM gets a fast launcher.
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

hidden_imports = (
    collect_submodules("statsmodels")
    + collect_submodules("xgboost")
    + collect_submodules("lightgbm")
    + ["openpyxl", "reportlab", "plotly", "kaleido"]
)
data_files = collect_data_files("plotly") + collect_data_files("openpyxl")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=data_files + [("data", "app/data")],
    hiddenimports=hidden_imports,
    excludes=[
        "torch", "darts", "pmdarima", "prophet", "pymc",
        "pytensor", "shap", "tensorflow",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="tomato-forecaster-lite",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    manifest="app.manifest" if False else None,  # supply Authenticode-signed manifest in CI
)
