"""Secure logging utilities for Gujarat Police CCTV Platform."""
import os
import re
import sys
import logging
from pathlib import Path
from typing import Any

# Patterns to sanitize from log messages
SENSITIVE_PATTERNS = [
    (re.compile(r'(password[:=]\s*["\']?)([^"\'\s&]+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(token[:=]\s*["\']?)([^"\'\s&]+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(cookie[:=]\s*["\']?)([^"\'\s;]+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(authorization[:=]\s*["\']?Bearer\s+)([^"\'\s]+)(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(Bearer\s+)[a-zA-Z0-9_\-\.]+', re.IGNORECASE), r'\1***REDACTED***'),
    (re.compile(r'([?&](?:token|password|auth|secret)=)([^&]+)', re.IGNORECASE), r'\1***REDACTED***'),
]


def mask_sensitive(text: str) -> str:
    """Mask sensitive credentials from any string."""
    if not isinstance(text, str):
        text = str(text)
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SensitiveFilter(logging.Filter):
    """Logging filter that scrubs sensitive credentials from record messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_sensitive(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: (mask_sensitive(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    (mask_sensitive(v) if isinstance(v, str) else v)
                    for v in record.args
                )
        return True


def get_logger(name: str = "cctv_platform", log_level: str = None) -> logging.Logger:
    """Configure and return a structured, sanitized logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sensitive_filter = SensitiveFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)
    logger.addHandler(console_handler)

    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "cctv.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(sensitive_filter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
