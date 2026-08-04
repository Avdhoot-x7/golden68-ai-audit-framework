"""Project-wide constants for the Golden 68 evaluation framework."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
RESULTS_DIR = DATA_DIR / "results"
REPORTS_DIR = DATA_DIR / "reports"

PILLARS = ("Causality", "Compliance", "Consistency")
PILLAR_SET = set(PILLARS)
PILLAR_COLORS = {
    "Causality": "#ff6b6b",
    "Compliance": "#4ecdc4",
    "Consistency": "#ffe66d",
}
PILLAR_DESCRIPTIONS = {
    "Causality": "Logical If-Then relationships",
    "Compliance": "EU AI Act mapping",
    "Consistency": "Behavior stability across prompt variants",
}

DEFAULT_BENCHMARK_DATASET_NAME = "golden68"
DEFAULT_BENCHMARK_DATASET_PATH = DATASET_DIR / "golden68.json"

# Reserved for future archival/source datasets. This path is intentionally not used
# in the evaluation pipeline so the benchmark dataset cannot be mixed accidentally.
ARCHIVAL_SOURCE_DATASET_NAME = "source_750"
ARCHIVAL_SOURCE_DATASET_PATH = DATASET_DIR / "archive" / "source_750.json"

RELIABILITY_PLACEHOLDER = {
    "metric": "Cohen's Kappa",
    "value": None,
}
