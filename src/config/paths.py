from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

NESTED_DUNNHUMBY_DIR = (
    PROJECT_ROOT
    / "dunnhumby_The-Complete-Journey"
    / "dunnhumby_The-Complete-Journey"
    / "dunnhumby_The-Complete-Journey CSV"
)

DOWNLOADS_DUNNHUMBY_DIR = (
    Path.home()
    / "Downloads"
    / "dunnhumby_The-Complete-Journey"
    / "dunnhumby_The-Complete-Journey"
    / "dunnhumby_The-Complete-Journey CSV"
)

DEFAULT_RAW_DIR_CANDIDATES = (
    RAW_DIR,
    NESTED_DUNNHUMBY_DIR,
    DOWNLOADS_DUNNHUMBY_DIR,
)


def resolve_raw_dir(raw_dir: str | Path | None = None) -> Path:
    """Return the first usable Dunnhumby CSV directory."""
    candidates: list[Path] = []
    if raw_dir:
        candidates.append(Path(raw_dir))
    if os.getenv("DUNNHUMBY_RAW_DIR"):
        candidates.append(Path(os.environ["DUNNHUMBY_RAW_DIR"]))
    candidates.extend(DEFAULT_RAW_DIR_CANDIDATES)

    for candidate in candidates:
        if (candidate / "transaction_data.csv").exists():
            return candidate.resolve()

    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find Dunnhumby CSV files. Set DUNNHUMBY_RAW_DIR or place "
        f"the CSVs in data/raw.\nChecked:\n{checked}"
    )


def ensure_project_dirs() -> None:
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        EXTERNAL_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

