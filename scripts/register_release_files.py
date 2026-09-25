"""Register added release files without accepting changes to existing evidence.

Run only when deliberately preparing a new release. This is not a verifier and
must not be used to claim independent provenance. Existing file hashes must all
match before any added documentation or tests can enter the manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    target = ROOT / "artifact_manifest.json"
    manifest = json.loads(target.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Existing release file changed: {name}")
    files = [p for p in ROOT.rglob("*") if p.relative_to(ROOT).parts[0] != ".git"]
    if any(p.is_symlink() for p in files):
        raise ValueError("Release must not contain symlinks")
    manifest["files"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(files) if p.is_file() and p != target}
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Registered {len(manifest['files'])} release files; previous hashes preserved.")


if __name__ == "__main__":
    main()
