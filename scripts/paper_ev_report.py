#!/usr/bin/env python3
"""
paper_ev_report.py — Corrected-paper EV report for llm_updown + over_below.

Only counts trades entered AFTER the 2026-06-11 realism fix (CLOB-ask entries +
2% fee), so the numbers reflect what live execution would actually do. Rows
before the cutoff used Gamma-mid entries and are excluded.

Usage:
    python scripts/paper_ev_report.py            # print to stdout
    python scripts/paper_ev_report.py --notify   # also push summary to ntfy
"""
import json, sys, os
from pathlib import Path

CUTOFF_TS = 1781139780   # 2026-06-11 01:03 UTC — realism fix deploy time
ROOT = Path(__file__).resolve().parent.parent

FILES = [
    ("llm_updown", ROOT / "data" / "llm_updown_paper.jsonl"),
    ("over_below", ROOT / "data" / "over_below_paper.jsonl"),
]


def _rows(path):
    if not path.exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _stats(rows):
    s = [r for r in rows
         if r.get("settled") and (r.get("entry_ts") or 0) >= CUTOFF_TS]
    if not s:
        return None
    wins = [r for r in s if (r.get("pnl") or 0) > 0]
    pnl = sum(r.get("pnl") or 0 for r in s)
    return {"n": len(s), "wr": 100 * len(wins) / len(s),
            "pnl": pnl, "ev": pnl / len(s), "rows": s}


def _by(rows, key):
    out = {}
    for r in rows:
        k = r.get(key, "?")
        out.setdefault(k, []).append(r)
    parts = []
    for k in sorted(out):
        g = out[k]
        w = [r for r in g if (r.get("pnl") or 0) > 0]
        p = sum(r.get("pnl") or 0 for r in g)
        parts.append("%s n=%d WR=%.0f%% $%.2f EV=%+.3f"
                     % (k, len(g), 100 * len(w) / len(g), p, p / len(g)))
    return parts


def main():
    notify = "--notify" in sys.argv
    lines = ["Corrected-paper EV (post-2026-06-11 fix, CLOB-ask + 2% fee)"]
    for name, path in FILES:
        st = _stats(_rows(path))
        if st is None:
            lines.append("\n[%s] no settled corrected-paper trades yet" % name)
            continue
        lines.append("\n[%s] n=%d  WR=%.1f%%  PnL=$%.2f  EV/trade=$%+.4f"
                     % (name, st["n"], st["wr"], st["pnl"], st["ev"]))
        for p in _by(st["rows"], "bet_direction"):
            lines.append("   dir " + p)
        if name == "llm_updown":
            for p in _by(st["rows"], "timeframe"):
                lines.append("   tf  " + p)

    report = "\n".join(lines)
    print(report)

    if notify:
        topic = None
        envp = ROOT / ".env"
        if envp.exists():
            for ln in open(envp, encoding="utf-8"):
                if ln.startswith("NTFY_TOPIC"):
                    topic = ln.split("=", 1)[1].strip()
                    break
        topic = topic or os.environ.get("NTFY_TOPIC")
        if topic:
            try:
                import httpx
                httpx.post("https://ntfy.sh/%s" % topic,
                           data=report.encode("utf-8"),
                           headers={"Title": "Paper EV report"}, timeout=10)
                print("\n[notified ntfy: %s]" % topic)
            except Exception as e:
                print("\n[ntfy failed: %s]" % e)
        else:
            print("\n[no NTFY_TOPIC — skipped notify]")


if __name__ == "__main__":
    main()
