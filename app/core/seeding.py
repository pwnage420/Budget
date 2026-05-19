"""Cross-library deterministic seeding.

Called once at process start and again inside every model worker so reproducibility
survives QThreadPool dispatch. The audit module captures the seed alongside every run.
"""
from __future__ import annotations

import os
import random


def seed_all(seed: int = 42) -> None:
    """Seed every library that has a global RNG. Quiet on libs that aren't installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
