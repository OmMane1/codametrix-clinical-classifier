"""Central config: paths, label space, and the Kaggle bootstrap mapping.

Edit ``HACKATHON_LABELS`` / ``KAGGLE_LABEL_MAP`` here rather than scattering
magic strings through the codebase.
"""
from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # gitignored: drop train.dat / test.dat here
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"          # gitignored: saved .joblib artifacts
REPORTS_DIR = ROOT / "reports"        # metrics, confusion matrices

# --- label space -----------------------------------------------------------
# The five categories the hidden test set is scored against.
HACKATHON_LABELS = [
    "Cardiology",
    "Neurology",
    "Orthopedics",
    "Gastroenterology",
    "Other",
]

# Best-effort mapping from the raw Kaggle `medical-text` numeric classes to the
# hackathon label space. NOTE: the Kaggle set has no Orthopedics examples, so a
# model trained purely on it can never predict that class. Use only to bootstrap
# / smoke-test until the organizers' cleaned ~1k-note set is available.
KAGGLE_LABEL_MAP = {
    "1": "Other",             # Neoplasms
    "2": "Gastroenterology",  # Digestive system diseases
    "3": "Neurology",         # Nervous system diseases
    "4": "Cardiology",        # Cardiovascular diseases
    "5": "Other",             # General pathological conditions
}

# --- MTSamples mapping -----------------------------------------------------
# Map MTSamples `medical_specialty` (whitespace-stripped) -> hackathon classes.
# Only the four target specialties are named; everything else falls to "Other".
MT_TARGET_MAP = {
    "Cardiovascular / Pulmonary": "Cardiology",
    "Neurology": "Neurology",
    "Orthopedic": "Orthopedics",
    "Gastroenterology": "Gastroenterology",
}

# Document-TYPE buckets in MTSamples — not specialties; they span every body
# system, so they're label-noise for a specialty classifier. Excluded from
# training when building "Other" from genuine specialties only.
MT_DOCTYPE_BUCKETS = {
    "Surgery", "Consult - History and Phy.", "Radiology", "General Medicine",
    "SOAP / Chart / Progress Notes", "Discharge Summary", "Emergency Room Reports",
    "Office Notes", "Letters", "IME-QME-Work Comp etc.", "Lab Medicine - Pathology",
    "Autopsy",
}

# --- defaults --------------------------------------------------------------
RANDOM_STATE = 42
DEFAULT_TEXT_COL = "text"
DEFAULT_LABEL_COL = "label"
