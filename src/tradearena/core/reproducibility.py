from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HASH_PREFIX = "sha256:"


def canonical_json(value: Any) -> str:
    """Serialize a value into stable JSON for reproducibility fingerprints."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return HASH_PREFIX + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return HASH_PREFIX + digest.hexdigest()


def compute_reproducibility_hash(payload: dict[str, Any]) -> str:
    """Hash the benchmark identity and audit metadata, excluding outcome metrics.

    Outcome metrics are intentionally excluded so the fingerprint represents the
    scenario, data, agent/config, redaction policy, and trajectory manifest
    rather than the score produced by that run.
    """

    stable_payload = {
        "schema_version": payload.get("schema_version"),
        "scenario_id": payload.get("scenario_id"),
        "agent": payload.get("agent", {}),
        "data_source": payload.get("data_source", {}),
        "execution_config": payload.get("execution_config", {}),
        "risk_config": payload.get("risk_config", {}),
        "trajectory_manifest": payload.get("trajectory_manifest", {}),
        "redaction": payload.get("redaction", {}),
    }
    return sha256_text(canonical_json(stable_payload))


def attach_reproducibility_hash(payload: dict[str, Any]) -> dict[str, Any]:
    updated = dict(payload)
    updated["reproducibility_hash"] = compute_reproducibility_hash(updated)
    return updated


def hash_trajectory_file(path: str | Path) -> dict[str, Any]:
    trajectory_path = Path(path)
    payload = _load_trajectory_json(trajectory_path)
    scenario_id = str(payload.get("experiment_name") or trajectory_path.stem)
    file_hash = sha256_file(trajectory_path)
    manifest = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "agent": {
            "agent_type": "trajectory_file",
            "model_family": "unknown_or_redacted",
            "model_identifier_redacted": True,
        },
        "data_source": {
            "name": "trajectory_file",
            "frequency": "unknown",
            "symbols": _symbols_from_trajectory(payload),
            "timestamp_policy": "as_recorded",
            "data_hash": file_hash,
        },
        "execution_config": {},
        "risk_config": {"risk_manager": "as_recorded", "risk_budget": {}},
        "trajectory_manifest": {
            "format": "json",
            # File name only, never the absolute path. Hashing the checkout
            # location made the reproducibility hash a function of where the
            # repository happened to live, so two reviewers with byte-identical
            # trajectories computed different values and the published hash
            # could not be reproduced by anyone but its author.
            "path_or_uri": trajectory_path.name,
            "raw_prompts_included": False,
            "raw_responses_included": False,
            "manifest_hash": file_hash,
        },
        "redaction": {
            "provider_secrets_removed": True,
            "timestamps_masked": False,
            "raw_provider_text_removed": True,
        },
    }
    return {
        "path": str(trajectory_path),
        "file_sha256": file_hash,
        "scenario_id": scenario_id,
        "reproducibility_hash": compute_reproducibility_hash(manifest),
    }


def _load_trajectory_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Trajectory file must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Trajectory file must be a JSON object")
    return payload


def _symbols_from_trajectory(payload: dict[str, Any]) -> list[str]:
    symbols: set[str] = set()
    for step in payload.get("steps", []):
        observation = step.get("observation", {}) if isinstance(step, dict) else {}
        prices = observation.get("prices", {}) if isinstance(observation, dict) else {}
        if isinstance(prices, dict):
            symbols.update(str(symbol) for symbol in prices)
    return sorted(symbols) or ["unknown"]
