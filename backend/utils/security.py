"""
MediSynth.AI — Security Utilities
Encryption, hashing, and data protection for HIPAA/GDPR compliance.
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
