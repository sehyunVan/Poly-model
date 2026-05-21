"""
scripts/record_outcomes.py

Fetch markets settled within a given time window from the Polymarket CLOB API
and write their outcomes (YES=1 / NO=0) to the feature cache so that
RollingTrainer can use them as training labels.

Intended to run daily via cron / Windows Task Scheduler, but can also be
invoked manually at any time.

Usage:
    python scripts/record_outcomes.py              # last 26 hours (default)
    python scripts/record_outcomes.py --hours 48   # last 48 hours
    python scripts/record_outcomes.py --dry-run    # report without writing
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from prediction.training import record_settled_outcomes


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record settled Polymarket outcomes into the feature cache."
    )
    p.add_argument(
        "--hours",
        type=float,
        default=26.0,
        help="Look-back window in hours (default: 26). "
             "26 h ensures yesterday's late-resolving markets are captured.",
    )
    p.add_argument(
        "--cache-dir",
        default="data/features_cache",
        help="Feature cache directory (default: data/features_cache).",
    )
    p.add_argument(
        "--clob-host",
        default="",
        help="Override CLOB API base URL (default: from CLOB_HOST env var "
             "or https://clob.polymarket.com).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse outcomes but do NOT write to the cache.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    cache_dir = ROOT / args.cache_dir

    print(
        f"[record_outcomes] Looking for markets settled since "
        f"{since.strftime('%Y-%m-%d %H:%M UTC')} "
        f"(last {args.hours:.0f} h)"
    )

    if args.dry_run:
        print("[record_outcomes] DRY-RUN mode — cache will NOT be modified.")

    if args.dry_run:
        # In dry-run mode we use a temp directory so nothing is written.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = record_settled_outcomes(
                since=since,
                clob_host=args.clob_host,
                cache_dir=tmp,
            )
    else:
        result = record_settled_outcomes(
            since=since,
            clob_host=args.clob_host,
            cache_dir=cache_dir,
        )

    # ── Report ────────────────────────────────────────────────────────────
    recorded = result["recorded"]
    skipped  = result["skipped"]
    api_err  = result["api_error"]

    if api_err:
        print("[record_outcomes] WARNING: API error occurred during fetch.")

    suffix = " (dry-run, not written)" if args.dry_run else ""
    print(
        f"[record_outcomes] Done — "
        f"recorded={recorded}{suffix}, skipped={skipped}, "
        f"api_error={api_err}"
    )

    if recorded == 0 and not api_err:
        print(
            "[record_outcomes] No new outcomes recorded. "
            "This is normal if no cached markets were settled in the window, "
            "or if bootstrap_training.py has not been run yet."
        )

    sys.exit(1 if api_err else 0)


if __name__ == "__main__":
    main()
