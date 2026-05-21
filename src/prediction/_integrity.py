"""
prediction/_integrity.py — model file integrity helpers.

Provides SHA-256 sidecar-based tamper detection for pickle model files.
Sidecar file naming convention: <model_path>.sha256
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

_log = logging.getLogger("prediction.integrity")


def save_hash(model_path: str | Path) -> None:
    """
    Compute SHA-256 of the saved model file and write it to a sidecar file.

    Call this immediately after saving a model file.
    The sidecar path is ``<model_path>.sha256``.
    """
    model_path = Path(model_path)
    data = model_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = Path(str(model_path) + ".sha256")
    sidecar.write_text(digest, encoding="utf-8")
    _log.debug("Hash saved: %s  sha256=%s…", model_path.name, digest[:16])


def verify_hash(model_path: str | Path) -> bool:
    """
    Verify the model file against its SHA-256 sidecar.

    Returns
    -------
    True  — file matches the stored hash, or no sidecar exists yet
            (legacy model without a hash — emits a WARNING).
    False — hash mismatch (file has been modified or corrupted).
    """
    model_path = Path(model_path)
    sidecar    = Path(str(model_path) + ".sha256")

    if not sidecar.exists():
        _log.warning(
            "No integrity hash found for %s — skipping verification. "
            "Run save() again to create a hash.",
            model_path.name,
        )
        return True  # allow loading legacy models

    expected = sidecar.read_text(encoding="utf-8").strip()
    actual   = hashlib.sha256(model_path.read_bytes()).hexdigest()

    if actual != expected:
        _log.error(
            "INTEGRITY VIOLATION: %s hash mismatch! "
            "expected=%s… actual=%s… — refusing to load.",
            model_path.name, expected[:16], actual[:16],
        )
        return False

    _log.debug("Integrity OK: %s", model_path.name)
    return True
