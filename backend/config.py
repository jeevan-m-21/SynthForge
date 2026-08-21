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
# Server & CORS
# ──────────────────────────────────────────────
HOST = os.getenv("SYNTH_HOST", "127.0.0.1")
PORT = int(os.getenv("SYNTH_PORT", "8000"))
DEBUG = os.getenv("SYNTH_DEBUG", "true").lower() == "true"
ENV = os.getenv("SYNTH_ENV", "development").lower()

_raw_cors = os.getenv(
    "SYNTH_CORS_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
)
CORS_ORIGINS = [orig.strip() for orig in _raw_cors.split(",") if orig.strip()]

# ──────────────────────────────────────────────
# Resource & Concurrency Limits (Phase 7B)
# ──────────────────────────────────────────────
MAX_SYNTH_ROWS = int(os.getenv("SYNTH_MAX_ROWS", "100000"))
MAX_EPOCHS = int(os.getenv("SYNTH_MAX_EPOCHS", "200"))
MAX_FL_ROUNDS = int(os.getenv("SYNTH_MAX_FL_ROUNDS", "20"))
MAX_EXECUTION_TIMEOUT_SECONDS = int(os.getenv("SYNTH_MAX_TIMEOUT_SECONDS", "300"))
MAX_ATTACK_SAMPLE_SIZE = int(os.getenv("SYNTH_MAX_ATTACK_SAMPLE_SIZE", "5000"))
MAX_CAT_COLS_FOR_ASSOC = int(os.getenv("SYNTH_MAX_CAT_COLS_FOR_ASSOC", "30"))
SYNTH_MAX_WORKERS = int(os.getenv("SYNTH_MAX_WORKERS", "4"))

# ──────────────────────────────────────────────
# Upload Limits
# ──────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.getenv("SYNTH_MAX_UPLOAD_SIZE_MB", "50"))

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
DEFAULT_EPOCHS = min(50, MAX_EPOCHS)
DEFAULT_BATCH_SIZE = 500
DEFAULT_NUM_ROWS = min(1000, MAX_SYNTH_ROWS)

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
_env_secret = os.getenv("SYNTH_ENCRYPTION_KEY")
if _env_secret:
    ENCRYPTION_KEY = _env_secret
elif ENV in ["production", "prod"]:
    raise RuntimeError("CRITICAL: SYNTH_ENCRYPTION_KEY environment variable must be set in production mode.")
else:
    ENCRYPTION_KEY = "synth-dev-key-local-only"

HASH_ALGORITHM = "sha256"
