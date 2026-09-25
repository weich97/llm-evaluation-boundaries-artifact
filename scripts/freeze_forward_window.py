"""Freeze a forward evaluation window before it opens.

Contamination control layer C2 (research plan 03): models cannot have
memorized a market window that has not happened yet. This script writes a
commitment file declaring the window, universe, frequency, and protocol, plus
a SHA-256 over the canonical declaration. Committing the file to git before
``window_start`` gives a publicly checkable timestamp; after the window
closes, the evaluation must run with exactly the declared settings.

Usage:

  python scripts/freeze_forward_window.py \
    --window-start 2026-07-01 --window-end 2026-09-30 \
    --symbols GSPC,BTC-USD,BTC=F --frequency weekly \
    --output docs/results/forward_window_commitment_2026q3.json

  python scripts/freeze_forward_window.py --verify \
    docs/results/forward_window_commitment_2026q3.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMITTED_FIELDS = (
    "window_start",
    "window_end",
    "symbols",
    "frequency",
    "execution_mode",
    "risk_profile",
    "rank_metric",
    "protocol_note",
)


def declaration_hash(declaration: dict) -> str:
    canonical = json.dumps(
        {field: declaration[field] for field in COMMITTED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify a forward evaluation window commitment.")
    parser.add_argument("--verify", metavar="PATH", default="", help="Verify an existing commitment file.")
    parser.add_argument("--window-start", default="")
    parser.add_argument("--window-end", default="")
    parser.add_argument("--symbols", default="GSPC,BTC-USD,BTC=F")
    parser.add_argument("--frequency", default="weekly", choices=["daily", "weekly"])
    parser.add_argument("--execution-mode", default="realistic-stress")
    parser.add_argument("--risk-profile", default="max-position-default")
    parser.add_argument("--rank-metric", default="sharpe")
    parser.add_argument(
        "--protocol-note",
        default="Forward window evaluated with the leaderboard matrix protocol pinned at the committed revision.",
    )
    parser.add_argument("--output", default="docs/results/forward_window_commitment.json")
    args = parser.parse_args(argv)

    if args.verify:
        path = Path(args.verify) if Path(args.verify).is_absolute() else ROOT / args.verify
        commitment = json.loads(path.read_text(encoding="utf-8"))
        expected = commitment.get("declaration_hash", "")
        actual = declaration_hash(commitment)
        if expected != actual:
            print(f"MISMATCH: file says {expected}, declaration hashes to {actual}", file=sys.stderr)
            return 1
        print(f"OK {path.name}: declaration hash verified ({actual})")
        return 0

    if not args.window_start or not args.window_end:
        parser.error("--window-start and --window-end are required when freezing")
    start = date.fromisoformat(args.window_start)
    end = date.fromisoformat(args.window_end)
    if end <= start:
        parser.error("--window-end must be after --window-start")
    today = datetime.now(timezone.utc).date()
    if start <= today:
        parser.error(
            f"window start {start} is not in the future (today is {today}); a forward commitment must precede the window"
        )

    declaration = {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "symbols": sorted(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()),
        "frequency": args.frequency,
        "execution_mode": args.execution_mode,
        "risk_profile": args.risk_profile,
        "rank_metric": args.rank_metric,
        "protocol_note": args.protocol_note,
    }
    commitment = {
        **declaration,
        "declaration_hash": declaration_hash(declaration),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_at_commit": _git_head(),
        "verification": (
            "Recompute with scripts/freeze_forward_window.py --verify <file>; the git history date of this"
            " file must precede window_start."
        ),
    }
    output_path = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(commitment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Froze forward window {start} -> {end} at {output_path}")
    print(f"Declaration hash: {commitment['declaration_hash']}")
    return 0


def _git_head() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
