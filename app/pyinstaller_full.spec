# PyInstaller spec — full build (includes PyTorch / darts / pymc).
#
# Build from the repo root:  pyinstaller app/pyinstaller_full.spec --clean --noconfirm
#
# Output WILL EXCEED the 250 MB single-exe cap because of PyTorch + PyMC.
# Use the lite spec for end-user distribution; this one is for analysts.
#
# Note: PyMC's PyTensor backend JIT-compiles C at runtime, which conflicts with a
# frozen environment. We force the Python fallback (pytensor cxx="") at app start
# when frozen — see app/main.py. BSTS will be slower in the .exe than under pip.
# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

hidden_imports = (
    collect_submodules("statsmodels")
    + collect_submodules("xgboost")
    + collect_submodules("lightgbm")
    + collect_submodules("prophet")
    + collect_submodules("pmdarima")
    + collect_submodules("shap")
    + ["openpyxl", "reportlab", "plotly", "kaleido", "scipy.special"]
)
# torch / darts / pymc are large; collect their submodules too but tolerate absence
for heavy in ("torch", "darts", "pymc", "pytensor"):
    try:
        hidden_imports += collect_submodules(heavy)
    except Exception:
        pass

data_files = (
    collect_data_files("plotly")
    + collect_data_files("prophet")
    + [(os.path.join(ROOT, "app", "data"), "app/data")]
)
for heavy in ("torch", "pymc", "pytensor"):
    try:
        data_files += collect_data_files(heavy)
    except Exception:
        pass

a = Analysis(
    [os.path.join(ROOT, "run_app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    excludes=["tensorflow"],
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
    name="tomato-forecaster-full",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
)
