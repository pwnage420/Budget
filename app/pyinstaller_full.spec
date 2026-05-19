# PyInstaller spec — full build (includes PyTorch / darts / pymc).
#
# Build with:  pyinstaller app/pyinstaller_full.spec
#
# Output will EXCEED the 250 MB single-exe cap because of PyTorch and PyMC.
# Use the lite spec for end-user distribution; this one is for analysts.
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

hidden_imports = (
    collect_submodules("statsmodels")
    + collect_submodules("xgboost")
    + collect_submodules("lightgbm")
    + collect_submodules("torch")
    + collect_submodules("darts")
    + collect_submodules("prophet")
    + collect_submodules("pmdarima")
    + collect_submodules("pymc")
    + collect_submodules("shap")
    + ["openpyxl", "reportlab", "plotly", "kaleido"]
)
data_files = (
    collect_data_files("plotly")
    + collect_data_files("openpyxl")
    + collect_data_files("torch")
    + collect_data_files("prophet")
)

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=data_files + [("data", "app/data")],
    hiddenimports=hidden_imports,
    excludes=["tensorflow"],
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
    name="tomato-forecaster-full",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
)
