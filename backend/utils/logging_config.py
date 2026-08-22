"""
MediSynth.AI — Structured Logging
"""
import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for audit trails."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry)


def setup_logging(level=logging.INFO):
    """Configure application-wide logging."""
    root_logger = logging.getLogger("MediSynth.AI")
    root_logger.setLevel(level)

    # Console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(f"MediSynth.AI.{name}")


SENSITIVE_LOG_KEYS = {"encryption_key", "secret", "password", "token", "key", "auth", "raw_data"}


def sanitize_log_dict(d: dict) -> dict:
    """Recursively scrub sensitive keys and protect logs from information leakage."""
    if not isinstance(d, dict):
        return d
    sanitized = {}
    for k, v in d.items():
        if any(sens in k.lower() for sens in SENSITIVE_LOG_KEYS):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [sanitize_log_dict(item) if isinstance(item, dict) else item for item in v]
        elif k.lower() in ["filename", "file_name"] and isinstance(v, str):
            from backend.utils.security import sanitize_filename
            sanitized[k] = sanitize_filename(v)
        else:
            sanitized[k] = v
    return sanitized


def audit_log(logger: logging.Logger, action: str, details: dict):
    """Log a privacy-sensitive operation for audit trail with sensitive value scrubbing."""
    safe_details = sanitize_log_dict(details)
    logger.info(
        f"AUDIT: {action}",
        extra={"extra_data": {"action": action, "details": safe_details}},
    )
