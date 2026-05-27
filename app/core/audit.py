"""Run-level audit trail — every forecast run gets a JSON record under
~/.tomato-forecaster/runs/<iso-timestamp>.json so the analyst can reproduce a board
forecast months later. Records only the data *hash*, not raw data values (security).
"""
from __future__ import annotations

import getpass
import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import __version__


def app_home() -> Path:
    home = Path.home() / ".tomato-forecaster"
    home.mkdir(parents=True, exist_ok=True)
    return home


def runs_dir() -> Path:
    d = app_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(df.to_csv(index=False).encode("utf-8"))
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for lib in ("numpy", "pandas", "scikit-learn", "statsmodels",
                "xgboost", "lightgbm", "prophet", "pmdarima"):
        try:
            mod = __import__(lib.replace("-", "_"))
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            continue
    return versions


@dataclass
class AuditRecord:
    timestamp: str
    app_version: str
    user: str
    platform: str
    data_sha256: str
    n_observations: int
    years: list[int]
    horizon: int
    models_attempted: list[str]
    models_succeeded: list[str]
    blend_strategy: str
    blend_weights: dict[str, float]
    blend_backtest_rmse: float
    library_versions: dict[str, str]
    seed: int
    extras: dict = field(default_factory=dict)


def make_record(history: pd.DataFrame, horizon: int, models_attempted: list[str],
                models_succeeded: list[str], blend_strategy: str,
                blend_weights: dict[str, float], blend_rmse: float,
                seed: int = 42, extras: dict | None = None) -> AuditRecord:
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        app_version=__version__,
        user=getpass.getuser(),
        platform=f"{platform.system()} {platform.release()}",
        data_sha256=data_hash(history),
        n_observations=int(len(history)),
        years=sorted(int(y) for y in history["year"].unique()),
        horizon=horizon,
        models_attempted=models_attempted,
        models_succeeded=models_succeeded,
        blend_strategy=blend_strategy,
        blend_weights={k: float(v) for k, v in blend_weights.items()},
        blend_backtest_rmse=float(blend_rmse) if blend_rmse == blend_rmse else None,
        library_versions=library_versions(),
        seed=seed,
        extras=extras or {},
    )


def write_record(record: AuditRecord, target_dir: Path | None = None) -> Path:
    target = (target_dir or runs_dir())
    target.mkdir(parents=True, exist_ok=True)
    ts = record.timestamp.replace(":", "-")
    path = target / f"{ts}.json"
    path.write_text(json.dumps(asdict(record), indent=2, default=str), encoding="utf-8")
    return path
