from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
)
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release.*",
)
logging.getLogger("torch.fx").setLevel(logging.ERROR)
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import METRICS_LOCK
from app.tasks import compute_metrics


def main() -> None:
    lock_path = Path(METRICS_LOCK)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("running")

    try:
        compute_metrics()
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
