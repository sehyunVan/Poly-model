"""
Auto-tune risk and signal parameters from virtual trading results.

auto_apply_suggestions() takes the output of
monitoring.feedback.suggest_parameter_adjustments() and writes accepted
changes directly to the config YAML files.

This is only called in VIRTUAL_MODE so that the bot can refine itself
safely without ever touching a real trading account.

Safety bounds prevent values from going outside sensible ranges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

# ── Tunable parameter registry ────────────────────────────────────────────────
# Maps param name → {file, key, min, max}
# "key" is the YAML key inside the file (may differ from the suggestion param).

TUNABLE_PARAMS: dict[str, dict] = {
    "alpha_threshold": {
        "file": "config/signal_params.yaml",
        "key":  "alpha_threshold",
        "min":  0.02,
        "max":  0.20,
    },
    "kelly_fraction": {
        "file": "config/risk_limits.yaml",
        "key":  "kelly_fraction",
        "min":  0.05,
        "max":  0.50,
    },
    "max_bet_pct": {
        "file": "config/risk_limits.yaml",
        "key":  "max_bet_pct",
        "min":  0.01,
        "max":  0.10,
    },
}


def auto_apply_suggestions(
    suggestions: list[dict],
    config_root: Union[str, Path] = ".",
) -> list[str]:
    """
    Apply float-valued parameter suggestions to the config YAML files.

    Args:
        suggestions:  List of dicts returned by suggest_parameter_adjustments().
                      Expected keys: {"param", "current", "suggested", "reason"}
        config_root:  Base directory that contains the config/ folder.
                      Defaults to current working directory.

    Returns:
        List of human-readable strings describing each applied change.
        Suggestions that are skipped (unknown param, non-float, out of bounds
        in an unexpected way, or failed I/O) are not included.

    Non-float suggestions (e.g. "model_retraining", "trading_enabled") are
    silently skipped — they require manual action or separate handling.
    """
    root   = Path(config_root)
    applied: list[str] = []

    # Group suggestions by config file to avoid repeated reads/writes
    file_changes: dict[str, dict] = {}   # file_path → {key: new_value}

    for s in suggestions:
        param     = s.get("param", "")
        suggested = s.get("suggested")
        reason    = s.get("reason", "")

        if param not in TUNABLE_PARAMS:
            continue   # non-float or unknown param — skip silently

        # Only handle numeric suggestions
        try:
            new_val = float(suggested)
        except (TypeError, ValueError):
            continue

        spec     = TUNABLE_PARAMS[param]
        clamped  = max(spec["min"], min(spec["max"], new_val))
        file_key = spec["file"]

        if file_key not in file_changes:
            file_changes[file_key] = {}
        file_changes[file_key][spec["key"]] = clamped

        applied.append(
            f"{param}: {s.get('current')} → {clamped:.4f} | {reason[:100]}"
        )

    # Write all changes for each file in one pass
    for rel_path, kv in file_changes.items():
        abs_path = root / rel_path
        try:
            if abs_path.exists():
                with open(abs_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                cfg = {}

            cfg.update(kv)

            with open(abs_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

        except Exception as exc:
            # Don't crash the midnight task over a config write failure
            print(f"[virtual.auto_tune] Failed to write {abs_path}: {exc}")
            # Remove failed entries from the applied list
            for key in kv:
                applied = [a for a in applied if not a.startswith(key)]

    return applied
