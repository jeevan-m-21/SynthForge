"""
SynthForge — Security Utilities
Encryption, hashing, and general data protection helpers.
"""
import hashlib
import hmac
import secrets
import base64
from typing import Optional


def hash_data(data: str, algorithm: str = "sha256") -> str:
    """Generate a cryptographic hash of data."""
    return hashlib.new(algorithm, data.encode("utf-8")).hexdigest()


def hash_record(record: dict, salt: Optional[str] = None) -> str:
    """Hash a data record for fingerprinting (non-reversible)."""
    canonical = "|".join(f"{k}={v}" for k, v in sorted(record.items()))
    if salt:
        canonical = f"{salt}:{canonical}"
    return hash_data(canonical)


def generate_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def generate_dataset_id() -> str:
    """Generate a unique dataset identifier."""
    return f"ds_{secrets.token_hex(8)}"


def generate_job_id() -> str:
    """Generate a unique job identifier."""
    return f"job_{secrets.token_hex(8)}"


def generate_federation_id() -> str:
    """Generate a unique federation identifier."""
    return f"fed_{secrets.token_hex(8)}"


def mask_pii(value: str, visible_chars: int = 3) -> str:
    """Mask personally identifiable information, showing only last N chars."""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


def compute_data_fingerprint(data_bytes: bytes) -> str:
    """Compute SHA-256 fingerprint of raw data for integrity verification."""
    return hashlib.sha256(data_bytes).hexdigest()


def constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def sanitize_filename(filename: Optional[str], default_name: str = "dataset.csv") -> str:
    """
    Sanitize an uploaded or external filename to protect against path traversal,
    null bytes, reserved device names, and illegal characters.
    """
    if not filename or not isinstance(filename, str):
        return default_name

    # 1. Reject null bytes
    cleaned = filename.replace("\x00", "")

    # 2. Extract base name across both POSIX and Windows path separators
    cleaned = cleaned.replace("\\", "/").split("/")[-1].strip()

    # 3. Strip leading traversal dots and spaces
    cleaned = cleaned.lstrip(". ")

    if not cleaned:
        return default_name

    # 4. Separate stem and extension
    if "." in cleaned:
        parts = cleaned.rsplit(".", 1)
        stem, ext = parts[0], "." + parts[1].lower()
    else:
        stem, ext = cleaned, ".csv"

    # 5. Sanitize stem to safe alphanumeric, underscore, hyphen
    safe_stem = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in stem)
    safe_stem = safe_stem.strip("._-")

    # 6. Ensure non-reserved Windows device name and non-empty stem
    if not safe_stem or safe_stem.upper() in RESERVED_WINDOWS_NAMES:
        safe_stem = f"file_{secrets.token_hex(4)}"

    # 7. Restrict extension to safe CSV extension
    if ext != ".csv":
        ext = ".csv"

    # 8. Limit length
    safe_stem = safe_stem[:80]
    return f"{safe_stem}{ext}"
