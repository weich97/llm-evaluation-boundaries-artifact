from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RedactionMode(str, Enum):
    PRIVATE_DEBUG = "private_debug"
    PUBLIC_ARTIFACT = "public_artifact"


_PROMPT_TEXT_KEYS = {
    "messages",
    "prompt",
    "raw_prompt",
    "raw_prompts",
    "system_prompt",
    "user_prompt",
}
_RESPONSE_TEXT_KEYS = {
    "provider_response",
    "provider_response_text",
    "raw_response",
    "raw_responses",
    "raw_response_text",
    "response",
    "response_text",
}
_SECRET_KEYS = {
    "access_token",
    "api_key",
    "api_secret",
    "apikey",
    "authorization",
    "bearer_token",
    "broker_password",
    "broker_token",
    "password",
    "refresh_token",
    "secret",
    "secret_key",
}
_ACCOUNT_KEYS = {"account", "account_email", "account_id", "account_number", "acct", "email"}
_TEXT_REDACTION_KEYS = {
    "agent_discussion_history",
    "rationale",
    "reflection",
    "risk_notes",
}

_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bAuthorization\s*[:=]\s*Bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{12,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?i)\b(account|acct)[-_ ]?(id|number|no)?\b\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{6,}"),
)

_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/](?:Users|TradeArena|TreadeArena|outputs)[\\/][^\s`\"'<>)]+"),
    re.compile(r"(?i)\b/(?:Users|home)/[^\s`\"'<>)]+"),
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_HASH_ONLY_PROMPT_CONTAINER_KEYS = {
    "prompt_template_id",
    "prompt_version",
    "prompt_sha256",
    "system_prompt_sha256",
    "raw_prompt_public",
}
_HASH_ONLY_RESPONSE_CONTAINER_KEYS = {
    "response_sha256",
    "response_format",
    "parse_status",
    "parse_error_class",
    "raw_response_public",
}


@dataclass(frozen=True)
class RedactionPolicy:
    """Controls how private LLM/debug text becomes public benchmark artifacts."""

    mode: RedactionMode = RedactionMode.PUBLIC_ARTIFACT
    hash_prefix: str = "sha256:"
    rationale_preview_chars: int = 0

    @classmethod
    def from_value(cls, value: RedactionPolicy | RedactionMode | str | None) -> RedactionPolicy:
        if isinstance(value, RedactionPolicy):
            return value
        if value is None:
            return PUBLIC_ARTIFACT_POLICY
        return cls(mode=RedactionMode(value))

    def redact(self, payload: Any) -> Any:
        if self.mode == RedactionMode.PRIVATE_DEBUG:
            return payload
        return self._redact_value(payload, parent_key="")

    def _redact_value(self, value: Any, parent_key: str) -> Any:
        key = _normalize_key(parent_key)
        if key in _TEXT_REDACTION_KEYS:
            if isinstance(value, list):
                return [self._redacted_text(item, parent_key) for item in value]
            return self._redacted_text(value, parent_key)
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for raw_key, raw_item in value.items():
                child_key = str(raw_key)
                normalized = _normalize_key(child_key)
                if normalized in _PROMPT_TEXT_KEYS:
                    redacted.setdefault("prompt_hash", self.hash_text(raw_item))
                    continue
                if normalized in _RESPONSE_TEXT_KEYS:
                    redacted.setdefault("response_hash", self.hash_text(raw_item))
                    continue
                if normalized in _SECRET_KEYS or normalized in _ACCOUNT_KEYS:
                    redacted[f"{child_key}_redacted"] = True
                    continue
                redacted[child_key] = self._redact_value(raw_item, child_key)
            return redacted
        if isinstance(value, list):
            return [self._redact_value(item, parent_key) for item in value]
        if isinstance(value, tuple):
            return [self._redact_value(item, parent_key) for item in value]
        if isinstance(value, str) and _looks_sensitive(value):
            return self._redacted_text(value, parent_key or "text")
        return value

    def _redacted_text(self, value: Any, label: str) -> str:
        text = _stable_text(value)
        digest = self.hash_text(text)
        if self.rationale_preview_chars <= 0:
            return f"[redacted {label} {digest}]"
        preview = text[: self.rationale_preview_chars]
        if _looks_sensitive(preview):
            preview = ""
        return f"{preview}[redacted {label} {digest}]"

    def hash_text(self, value: Any) -> str:
        text = _stable_text(value)
        return self.hash_prefix + hashlib.sha256(text.encode("utf-8")).hexdigest()


PUBLIC_ARTIFACT_POLICY = RedactionPolicy(mode=RedactionMode.PUBLIC_ARTIFACT)
PRIVATE_DEBUG_POLICY = RedactionPolicy(mode=RedactionMode.PRIVATE_DEBUG)


def redact_public_artifact(payload: Any) -> Any:
    return PUBLIC_ARTIFACT_POLICY.redact(payload)


def scan_public_artifact_payload(payload: Any) -> list[str]:
    findings: list[str] = []
    _scan_value(payload, "$", findings)
    return findings


def scan_public_artifact_paths(paths: list[str | Path]) -> list[str]:
    findings: list[str] = []
    for root in paths:
        path = Path(root)
        if not path.exists():
            findings.append(f"{path}: path does not exist")
            continue
        candidates = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for candidate in candidates:
            if {"llm_cache", "poe_skill_task_answers", "poe_skill_task_answers_smoke"} & set(candidate.parts):
                continue
            if candidate.suffix.lower() not in {".json", ".jsonl", ".md", ".html", ".txt"}:
                continue
            findings.extend(_scan_file(candidate))
    return findings


def _scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    findings = [f"{path}: sensitive text pattern: {pattern.pattern}" for pattern in _SECRET_VALUE_PATTERNS if pattern.search(text)]
    findings.extend(
        f"{path}: local filesystem path pattern: {pattern.pattern}" for pattern in _LOCAL_PATH_PATTERNS if pattern.search(text)
    )
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            findings.append(f"{path}: invalid JSON: {exc}")
        else:
            findings.extend(f"{path}: {finding}" for finding in scan_public_artifact_payload(payload))
    elif path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                findings.append(f"{path}:{line_number}: invalid JSONL row: {exc}")
                continue
            findings.extend(f"{path}:{line_number}: {finding}" for finding in scan_public_artifact_payload(payload))
    else:
        raw_field_pattern = re.compile(
            r'(?i)"(prompt|raw_prompt|raw_response|response_text|authorization|api_key|account_email)"\s*:'
        )
        if raw_field_pattern.search(text):
            findings.append(f"{path}: forbidden raw public-artifact field")
    return findings


def _scan_value(value: Any, path: str, findings: list[str]) -> None:
    if isinstance(value, dict):
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            normalized = _normalize_key(key)
            if normalized in _PROMPT_TEXT_KEYS | _RESPONSE_TEXT_KEYS and not _is_hash_only_manifest_container(
                normalized, raw_item
            ):
                findings.append(f"{path}.{key}: raw prompt/response field is not allowed")
            if normalized in _SECRET_KEYS:
                findings.append(f"{path}.{key}: secret field is not allowed")
            if normalized in _ACCOUNT_KEYS:
                findings.append(f"{path}.{key}: account or email field is not allowed")
            _scan_value(raw_item, f"{path}.{key}", findings)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_value(item, f"{path}[{index}]", findings)
    elif isinstance(value, str):
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(value):
                findings.append(f"{path}: sensitive text pattern: {pattern.pattern}")


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _is_hash_only_manifest_container(normalized_key: str, value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    normalized_keys = {_normalize_key(str(key)) for key in value}
    if normalized_key in _PROMPT_TEXT_KEYS:
        return (
            normalized_keys <= _HASH_ONLY_PROMPT_CONTAINER_KEYS
            and value.get("raw_prompt_public") is False
            and _is_sha256(value.get("prompt_sha256"))
        )
    if normalized_key in _RESPONSE_TEXT_KEYS:
        return (
            normalized_keys <= _HASH_ONLY_RESPONSE_CONTAINER_KEYS
            and value.get("raw_response_public") is False
            and _is_sha256(value.get("response_sha256"))
        )
    return False


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)
    except TypeError:
        return str(value)


def _looks_sensitive(value: str) -> bool:
    return any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS)
