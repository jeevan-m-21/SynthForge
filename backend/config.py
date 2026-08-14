"""
MediSynth.AI — Global Configuration
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
MODELS_DIR = STORAGE_DIR / "models"
GENERATED_DIR = STORAGE_DIR / "generated"
REPORTS_DIR = STORAGE_DIR / "reports"
FRONTEND_DIR = BASE_DIR / "frontend"

# Create directories
for d in [DATA_DIR, UPLOAD_DIR, MODELS_DIR, GENERATED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("SYNTH_HOST", "127.0.0.1")
PORT = int(os.getenv("SYNTH_PORT", "8000"))
DEBUG = os.getenv("SYNTH_DEBUG", "true").lower() == "true"

# ──────────────────────────────────────────────
# Differential Privacy Defaults
# ──────────────────────────────────────────────
DEFAULT_EPSILON = float(os.getenv("SYNTH_EPSILON", "1.0"))
DEFAULT_DELTA = float(os.getenv("SYNTH_DELTA", "1e-5"))
MAX_EPSILON_BUDGET = float(os.getenv("SYNTH_MAX_EPSILON", "10.0"))
PRIVACY_WARNING_THRESHOLDS = [0.5, 0.75, 0.9]  # warn at 50%, 75%, 90%

# ──────────────────────────────────────────────
# Generator Defaults
# ──────────────────────────────────────────────
DEFAULT_MODEL_TYPE = "ctgan"  # ctgan | tvae | gaussian_copula
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 500
DEFAULT_NUM_ROWS = 1000

# ──────────────────────────────────────────────
# ML Validation
# ──────────────────────────────────────────────
ML_TEST_SPLIT = 0.2
ML_RANDOM_STATE = 42
ML_N_ESTIMATORS = 100

# ──────────────────────────────────────────────
# Federated Learning
# ──────────────────────────────────────────────
FL_DEFAULT_ROUNDS = 5
FL_DEFAULT_LOCAL_EPOCHS = 50
FL_MIN_HOSPITALS = 2

# ──────────────────────────────────────────────
# Security
# ──────────────────────────────────────────────
ENCRYPTION_KEY = os.getenv("SYNTH_ENCRYPTION_KEY", "synth-health-guard-default-key-change-in-prod")
HASH_ALGORITHM = "sha256"
