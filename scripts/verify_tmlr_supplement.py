"""Fail-closed, offline reproduction of the complete released TMLR artifact.

Run from a fresh extraction. Original files are never modified. Figure rendering
requires the separately listed dependencies; absence is an error, not a skip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
A = "docs/results/v0_3_fixed_intent_replay"
B = "docs/results/finaudit_multilabel"
REPRODUCED = (
    *(f"{A}/{n}.csv" for n in ("factorial_estimands", "factorial_by_model_scenario",
                              "ranking_stability", "response_path_divergence")),
    *(f"{B}/{n}.csv" for n in ("primary_violation_drop", "condition_metrics", "model_overall",
                              "coverage_decomposition")),
    *(f"docs/results/finaudit/{n}.csv" for n in ("corrected_key_rescore", "intervention_v2", "self_audit_v2")),
    "paper/tech_report/appendix_tables.tex",
    *(f"paper/tech_report/figures/{n}.pdf" for n in (
        "execution_identification", "audit_cardinality", "pooling_hides_structure")),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root: Path) -> None:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tmlr.ab.artifact.v2":
        raise ValueError("Unexpected artifact schema")
    if manifest.get("reproduced") != list(REPRODUCED):
        raise ValueError("Required reproduction contract differs")
    expected = manifest["files"]
    if not expected or not isinstance(expected, dict):
        raise ValueError("Empty file manifest")
    for name, sha in expected.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or path.as_posix() != name or "\\" in name
                or any(p in {".", ".."} or ":" in p for p in path.parts)):
            raise ValueError(f"Unsafe manifest path: {name}")
        if (not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise ValueError(f"Invalid file hash: {name}")
    if len({name.casefold() for name in expected}) != len(expected):
        raise ValueError("Case-insensitive path collision")
    if not set(REPRODUCED).issubset(expected):
        raise ValueError("Required reproduced outputs missing from manifest")
    # A public checkout contains Git administration files, not research inputs.
    # Only the root metadata directory/file is exempt; nested extras still fail.
    paths = [p for p in root.rglob("*") if p.relative_to(root).parts[0] != ".git"]
    if any(p.is_symlink() for p in paths):
        raise ValueError("Symlinks are not allowed")
    actual = {p.relative_to(root).as_posix() for p in paths
              if p.is_file() and p != root / "artifact_manifest.json"}
    if actual != set(expected):
        raise ValueError(f"File-set mismatch: missing={sorted(set(expected)-actual)[:3]}, "
                         f"extra={sorted(actual-set(expected))[:3]}")
    for name, sha in expected.items():
        if digest(root / name) != sha:
            raise ValueError(f"Hash mismatch: {name}")


def deny_network(event: str, args: object) -> None:
    if event in {"socket.connect", "socket.getaddrinfo", "socket.bind"}:
        raise RuntimeError("Network access is forbidden during artifact verification")


def run(root: Path, script: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(root / ".render_config")
    command = [sys.executable, "-B", str(root / "scripts/verify_tmlr_supplement.py"),
               "--worker", script, *args]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Failed {script}:\n{result.stdout}\n{result.stderr}")
    print(f"PASS {script}", flush=True)


def reconstruct_primary_cache(root: Path) -> None:
    sys.path.insert(0, str(root / "scripts"))
    import run_multilabel_audit_eval as runner

    results = [json.loads(x) for x in (root / "outputs/audit_multilabel_eval/multilabel_audit_results.jsonl")
               .read_text(encoding="utf-8").splitlines() if x]
    raw = [json.loads(x) for x in (root / "evidence/primary_responses.jsonl")
           .read_text(encoding="utf-8").splitlines() if x]
    keys = [(r["model"], r["task_id"], r["sample"]) for r in raw]
    if len(keys) != 600 or len(set(keys)) != 600:
        raise ValueError("Raw response grid must contain exactly 600 unique keys")
    bykey = dict(zip(keys, raw, strict=True))
    manifest = json.loads((root / "outputs/audit_multilabel_tasks/manifest.json").read_text(encoding="utf-8"))
    domains = {r["task_id"]: r["domain"] for r in manifest["tasks"]}
    records = []
    result_keys = set()
    for result in results:
        key = (result["model"], result["task_id"], result["sample"])
        if key in result_keys or key not in bykey:
            raise ValueError("Missing/duplicate primary result")
        result_keys.add(key)
        response = bykey[key]
        if (response["response_sha256"] != result["response_sha256"]
                or response["request_hash"] != result["request_hash"]
                or hashlib.sha256(response["response_text"].encode()).hexdigest() != result["response_sha256"]):
            raise ValueError("Raw response/result binding mismatch")
        provider, model = result["model"].split(":", 1)
        prompt = runner.build_prompt(root / "outputs/audit_multilabel_tasks/tasks" / result["task_id"],
                                     domains[result["task_id"]])
        records.append({**{k: result[k] for k in ("cache_key", "task_id", "sample", "prompt_hash",
                                                "request_hash", "decoding_hash")},
                        "provider": provider, "model": model, "prompt": prompt,
                        "response_text": response["response_text"]})
    if result_keys != set(bykey):
        raise ValueError("Primary raw/results grid differs")
    cache = root / ".verification_cache"
    cache.mkdir()
    (cache / "auditml_reconstructed.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def reproduce(root: Path) -> None:
    verify_manifest(root)
    with tempfile.TemporaryDirectory(prefix="tmlr-verify-") as temp:
        work = Path(temp) / "artifact"
        shutil.copytree(root, work, ignore=lambda src, names: (
            {".git"} if Path(src) == root and ".git" in names else set()))
        run(work, "scripts/verify_tmlr_supplement.py", "--prepare-cache")
        run(work, "scripts/analyze_v03_fixed_intent_replay.py")
        run(work, "scripts/analyze_multilabel_audit.py", "--cache-dir", ".verification_cache")
        run(work, "scripts/tmlr_evidence.py")
        run(work, "scripts/build_appendix_tables.py")
        run(work, "scripts/verify_report_claims.py", "--strict")
        for script in ("render_studyA_figures", "render_finaudit_figures", "render_pooling_figures"):
            run(work, f"scripts/{script}.py")
        names = REPRODUCED
        for name in names:
            # Text outputs may acquire platform-native CSV newlines.
            a, b = (root / name).read_bytes(), (work / name).read_bytes()
            if name.endswith(".pdf"):
                same = a == b
            else:
                same = a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
            if not same:
                raise ValueError(f"Recomputed output differs: {name}")
        print(f"PASS {len(names)} regenerated tables, appendix and figures match; no skipped checks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.addaudithook(deny_network)
    if args.worker:
        script, *rest = args.worker
        target = (ROOT / script).resolve()
        if target.parent != (ROOT / "scripts").resolve() or target.suffix != ".py":
            raise ValueError("Worker must be a packaged analysis script")
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.argv = [str(target), *rest]
        runpy.run_path(str(target), run_name="__main__")
    elif args.prepare_cache:
        reconstruct_primary_cache(ROOT)
    else:
        reproduce(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
