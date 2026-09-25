from __future__ import annotations

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tradearena.core.domain import MarketSnapshot, PortfolioState, Signal


@dataclass
class DeepSeekLLMAnalyst:
    """Chat-completions analyst that emits structured trading signals with response caching."""

    model: str = "deepseek-v4-flash"
    cache_path: str = "data/llm_cache/deepseek_analyst.jsonl"
    api_key_env: str = "DEEPSEEK_API_KEY"
    fallback_api_key_env: str = ""
    api_base_url: str = "https://api.deepseek.com"
    provider: str = "deepseek"
    api_model: str = ""
    api_protocol: str = "openai_chat_completions"
    thinking: str = "disabled"
    use_response_format: bool = True
    require_api_key: bool = True
    timeout_seconds: int = 60
    max_retries: int = 5
    use_risk_feedback: bool = True
    risk_feedback_mode: str = "true"
    output_mode: str = "rationale"
    mask_timestamps: bool = False
    anonymize_symbols: bool = False
    sample_index: int = 0
    temperature: float = 0.0
    # When set, replaces the trading-analyst system message verbatim (no
    # response-shape suffix). Used by non-trading callers (e.g. audit evals)
    # that reuse this transport but must not inherit the analyst role.
    system_prompt_override: str | None = None
    name: str = "deepseek-llm-analyst"
    _cache_entries: dict[str, dict[str, Any]] | None = field(default=None, init=False, repr=False)
    _cache_mtime_ns: int | None = field(default=None, init=False, repr=False)
    _call_trace: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def analyze(self, snapshot: MarketSnapshot, portfolio: PortfolioState, memory: object) -> list[Signal]:
        alias_by_symbol = self._symbol_aliases(snapshot)
        symbol_by_alias = {alias: symbol for symbol, alias in alias_by_symbol.items()}
        prompt = self._prompt(snapshot, portfolio, memory)
        if alias_by_symbol:
            prompt = _replace_quoted_symbols(prompt, alias_by_symbol)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        # sample_index=0 keeps the legacy key format so existing caches replay.
        cache_key = f"{self.provider}:{self.model}:{prompt_hash}"
        if self.sample_index:
            cache_key = f"{cache_key}:s{int(self.sample_index)}"
        cached = self._cache().get(cache_key)
        started = time.time()
        if cached is None:
            response_text = self._call_deepseek(prompt)
            latency_ms = (time.time() - started) * 1000
            cached = {
                "cache_key": cache_key,
                "model": self.model,
                "api_model": self.api_model or self.model,
                "provider": self.provider,
                "prompt_hash": prompt_hash,
                "prompt": prompt,
                "response_text": response_text,
                "latency_ms": latency_ms,
                "sample_index": int(self.sample_index),
                "created_at": int(time.time()),
            }
            self._append_cache(cached)
        else:
            response_text = str(cached["response_text"])
            latency_ms = 0.0

        response_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        self._call_trace.append(
            {
                "prompt_hash": prompt_hash,
                "response_hash": response_hash,
                "sample_index": int(self.sample_index),
                "cache_hit": latency_ms == 0.0,
            }
        )
        parsed = self._parse_response(response_text)
        signals = []
        for item in self._signal_items(parsed):
            symbol = str(item.get("symbol", ""))
            symbol = symbol_by_alias.get(symbol, symbol)
            if symbol not in snapshot.bars:
                continue
            target_weight = _to_float(item.get("target_weight", item.get("weight", "")), None)
            score = (
                max(-1.0, min(1.0, target_weight / 5.0))
                if target_weight is not None and self.output_mode == "weights_only"
                else max(-1.0, min(1.0, _to_float(item.get("score", 0.0), 0.0) or 0.0))
            )
            signals.append(
                Signal(
                    symbol=symbol,
                    score=score,
                    horizon=str(item.get("horizon", "1w")),
                    confidence=max(0.0, min(1.0, _to_float(item.get("confidence", 1.0), 1.0) or 0.0)),
                    rationale="" if self.output_mode == "weights_only" else str(item.get("rationale", "")),
                    metadata={
                        "analyst": self.name,
                        "feature": "llm_market_reasoning",
                        "llm_call": True,
                        "llm_provider": self.provider,
                        "model": self.model,
                        "api_model": self.api_model or self.model,
                        "llm_output_mode": self.output_mode,
                        "prompt_hash": prompt_hash,
                        "response_hash": response_hash,
                        "latency_ms": latency_ms,
                        "sample_index": int(self.sample_index),
                        "risk_feedback_mode": self.risk_feedback_mode if self.use_risk_feedback else "hidden",
                        "timestamp_masked": self.mask_timestamps,
                        "symbols_anonymized": self.anonymize_symbols,
                        "risk_notes": "" if self.output_mode == "weights_only" else str(item.get("risk_notes", "")),
                        "target_weight": target_weight if target_weight is not None else "",
                    },
                )
            )
        return signals

    def _prompt(self, snapshot: MarketSnapshot, portfolio: PortfolioState, memory: object = None) -> str:
        bars = []
        for symbol, bar in snapshot.bars.items():
            intraperiod_return = (bar.close / bar.open) - 1.0 if bar.open else 0.0
            bars.append(
                {
                    "symbol": symbol,
                    "open": round(bar.open, 4),
                    "high": round(bar.high, 4),
                    "low": round(bar.low, 4),
                    "close": round(bar.close, 4),
                    "volume": round(bar.volume, 2),
                    "intraperiod_return": round(intraperiod_return, 6),
                }
            )
        payload = {
            "timestamp": self._prompt_timestamp(snapshot, memory),
            "bars": bars,
            "portfolio": {
                "cash": round(portfolio.cash, 2),
                "positions": portfolio.positions,
                "equity": round(portfolio.equity(), 2),
            },
            "task": (
                "Return one target weight per symbol for a one-week trading horizon. "
                "Do not include rationales, reflections, narrative explanations, or risk notes."
                if self.output_mode == "weights_only"
                else "Return one signal per symbol for a one-week trading horizon."
            ),
            "scale": (
                "target_weight is long-only in [0, 0.8]; use 0 for cash or no position."
                if self.output_mode == "weights_only"
                else "score is in [-1, 1], where positive means bullish and negative means bearish."
            ),
            "output_mode": self.output_mode,
        }
        if self.use_risk_feedback:
            if self.risk_feedback_mode == "placebo":
                recent_feedback = _placebo_recent_risk_feedback(memory)
                long_term_memory = _placebo_long_term_risk_memory(memory, limit=52)
            elif self.risk_feedback_mode == "contrarian":
                recent_feedback = _contrarian_recent_risk_feedback(memory)
                long_term_memory = _contrarian_long_term_risk_memory(memory, limit=52)
            else:
                recent_feedback = _recent_risk_feedback(memory)
                long_term_memory = _long_term_risk_memory(memory, limit=52)
            if self.risk_feedback_mode == "neutral":
                # Ablation: same recalled data, no behavioral directive --
                # separates "reacts to fabricated evidence" from "obeys the
                # conservatism instruction".
                instruction = (
                    "Recent risk feedback and long-term risk memory are "
                    "provided for reference."
                )
            else:
                instruction = (
                    "If recent decisions were clipped, blocked, rejected, or caused risk violations, "
                    "revise the next signal strength and confidence conservatively unless the new market evidence is strong. "
                    "Use the long-term risk memory to avoid repeating exposure patterns that were clipped, rejected, or violated risk constraints."
                )
            payload.update(
                {
                    "recent_risk_feedback": recent_feedback,
                    "long_term_risk_memory": long_term_memory,
                    "risk_feedback_instruction": instruction,
                }
            )
        else:
            payload["risk_feedback_instruction"] = "Risk feedback is intentionally hidden for this ablation."
        return json.dumps(payload, sort_keys=True)

    def _symbol_aliases(self, snapshot: MarketSnapshot) -> dict[str, str]:
        """Stable symbol -> ASSET_NN aliases for contamination-controlled prompts.

        Aliases follow the sorted symbol order, so a fixed universe maps the
        same way on every step of a run.
        """

        if not self.anonymize_symbols:
            return {}
        return {
            symbol: f"ASSET_{index:02d}"
            for index, symbol in enumerate(sorted(snapshot.bars), start=1)
        }

    def _prompt_timestamp(self, snapshot: MarketSnapshot, memory: object = None) -> str:
        if not self.mask_timestamps:
            return snapshot.timestamp.isoformat()
        events = getattr(memory, "events", []) if memory is not None else []
        return f"T+{len(events)}"

    def _call_deepseek(self, prompt: str) -> str:
        if self.api_protocol == "cache_only":
            raise RuntimeError(
                f"{self.provider} analyst is configured for cache-only replay. "
                f"No cached response was found for model={self.model}; live {self.provider} API calls are not enabled by this adapter."
            )
        api_key = _get_secret(self.api_key_env) or _get_secret(self.fallback_api_key_env)
        if not api_key and self.require_api_key:
            fallback = f" or {self.fallback_api_key_env}" if self.fallback_api_key_env else ""
            raise RuntimeError(
                f"{self.api_key_env}{fallback} is not set and no cached LLM response is available."
            )
        if self.output_mode == "weights_only":
            response_shape = (
                '{"weights":[{"symbol":"SYMBOL","target_weight":0.0,'
                '"confidence":0.0,"horizon":"1w"}]}.'
            )
        else:
            response_shape = (
                '{"signals":[{"symbol":"SYMBOL","score":0.0,"confidence":0.0,'
                '"horizon":"1w","rationale":"short reason","risk_notes":"short risk note"}]}.'
            )
        system_content = self.system_prompt_override or (
            "You are a cautious trading research analyst. "
            "Use only the provided OHLCV and portfolio state. "
            "Return calibrated signals or target weights, not executable orders. Do not mention API keys. "
            "Return only valid JSON with this shape: "
            f"{response_shape}"
        )
        request_body = {
            "model": self.api_model or self.model,
            "temperature": float(self.temperature),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
        }
        if self.use_response_format:
            request_body["response_format"] = {"type": "json_object"}
        if self.thinking:
            request_body["thinking"] = {"type": self.thinking}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{self.api_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(2.0 * attempt)
                    continue
                raise RuntimeError(f"{self.provider} API error {exc.code}; response body omitted to avoid leaking secrets.") from exc
            except (http.client.IncompleteRead, http.client.RemoteDisconnected, urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(f"{self.provider} API transport error after {attempt} attempts: {type(exc).__name__}") from exc
                time.sleep(1.5 * attempt)
        return str(payload["choices"][0]["message"]["content"])

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        text = response_text.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    parsed = {"signals": []}
            else:
                parsed = {"signals": []}
        return parsed if isinstance(parsed, dict) else {"signals": []}

    def _signal_items(self, parsed: dict[str, Any]) -> list[dict[str, Any]]:
        if self.output_mode != "weights_only":
            items = parsed.get("signals", [])
        else:
            items = parsed.get("weights") or parsed.get("target_weights") or parsed.get("signals", [])
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    def _cache(self) -> dict[str, dict[str, Any]]:
        path = Path(self.cache_path)
        if not path.exists():
            self._cache_entries = {}
            self._cache_mtime_ns = None
            return {}
        mtime_ns = path.stat().st_mtime_ns
        if self._cache_entries is not None and self._cache_mtime_ns == mtime_ns:
            return self._cache_entries
        cache: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                cache[str(item["cache_key"])] = item
                cache[_normalized_cache_key(item, self.provider)] = item
        self._cache_entries = cache
        self._cache_mtime_ns = mtime_ns
        return cache

    def _append_cache(self, item: dict[str, Any]) -> None:
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
        if self._cache_entries is None:
            self._cache_entries = {}
        self._cache_entries[str(item["cache_key"])] = item
        self._cache_entries[_normalized_cache_key(item, self.provider)] = item
        self._cache_mtime_ns = path.stat().st_mtime_ns


def _normalized_cache_key(item: dict[str, Any], default_provider: str) -> str:
    """Fallback lookup key for legacy entries whose cache_key lacks the provider prefix.

    Repeated-sample entries keep their ``:s<n>`` suffix so they never shadow the
    sample-0 entry for the same prompt.
    """

    key = f"{item.get('provider', default_provider)}:{item.get('model', '')}:{item.get('prompt_hash', '')}"
    tail = str(item.get("cache_key", "")).rsplit(":", 1)[-1]
    if len(tail) > 1 and tail.startswith("s") and tail[1:].isdigit():
        key = f"{key}:{tail}"
    return key


def _replace_quoted_symbols(serialized_prompt: str, alias_by_symbol: dict[str, str]) -> str:
    """Replace every quoted symbol occurrence in a JSON prompt with its alias.

    Operating on the serialized JSON catches symbols wherever they appear
    (bars, positions, recalled risk feedback). Longest symbols are replaced
    first so overlapping tickers cannot corrupt each other, and only quoted
    occurrences are touched so ordinary words stay intact.
    """

    for symbol in sorted(alias_by_symbol, key=len, reverse=True):
        serialized_prompt = serialized_prompt.replace(f'"{symbol}"', f'"{alias_by_symbol[symbol]}"')
    return serialized_prompt


def _get_secret(name: str) -> str:
    if not name:
        return ""
    value = os.environ.get(name, "")
    if value:
        return _clean_secret(value)
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return _clean_secret(str(value))
    except OSError:
        return ""


def _clean_secret(value: str) -> str:
    cleaned = value.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _to_float(value: object, default: float | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


ChatCompletionsLLMAnalyst = DeepSeekLLMAnalyst


def _recalled_step_events(memory: object, limit: int) -> list[Any]:
    """Recall step events through ``recent`` so memory decorators (for example
    read-time pollution) apply to the prompt; fall back to the raw journal for
    stores without a ``recent`` method."""

    recent_fn = getattr(memory, "recent", None) if memory is not None else None
    if callable(recent_fn):
        try:
            return list(recent_fn("step", limit))
        except TypeError:
            pass
    events = getattr(memory, "events", []) if memory is not None else []
    return list(events[-limit:])


def _recent_risk_feedback(memory: object) -> list[dict[str, Any]]:
    feedback = []
    for event in _recalled_step_events(memory, 3):
        payload = getattr(event, "payload", None)
        if payload is None and isinstance(event, dict):
            payload = event.get("payload", event)
        if not isinstance(payload, dict):
            continue
        risk_report = payload.get("risk_report")
        execution_report = payload.get("execution_report")
        equity = payload.get("equity")
        risk_report = _object_to_dict(risk_report)
        execution_report = _object_to_dict(execution_report)
        feedback.append(
            {
                "clipped_count": risk_report.get("clipped_count", 0),
                "blocked_count": risk_report.get("blocked_count", 0),
                "risk_violations": len(payload.get("risk_violations", []) or []),
                "rejected_orders": execution_report.get("rejected_orders", 0),
                "pending_orders": execution_report.get("pending_orders", 0),
                "total_slippage": round(float(execution_report.get("total_slippage", 0.0) or 0.0), 4),
                "equity": round(float(equity), 2) if isinstance(equity, (int, float)) else None,
            }
        )
    return feedback


def _long_term_risk_memory(memory: object, limit: int = 52) -> dict[str, Any]:
    summaries = [_risk_event_summary(event) for event in _recalled_step_events(memory, limit)]
    summaries = [summary for summary in summaries if summary]
    if not summaries:
        return {
            "lookback_steps": 0,
            "risk_gate_rate": 0.0,
            "avg_clipped_count": 0.0,
            "avg_blocked_count": 0.0,
            "avg_risk_violations": 0.0,
            "avg_rejected_orders": 0.0,
            "avg_total_slippage": 0.0,
            "recent_failure_examples": [],
        }
    risk_events = [
        item
        for item in summaries
        if item["clipped_count"] + item["blocked_count"] + item["risk_violations"] + item["rejected_orders"] > 0
    ]
    return {
        "lookback_steps": len(summaries),
        "risk_gate_rate": round(len(risk_events) / len(summaries), 4),
        "avg_clipped_count": round(_avg(item["clipped_count"] for item in summaries), 4),
        "avg_blocked_count": round(_avg(item["blocked_count"] for item in summaries), 4),
        "avg_risk_violations": round(_avg(item["risk_violations"] for item in summaries), 4),
        "avg_rejected_orders": round(_avg(item["rejected_orders"] for item in summaries), 4),
        "avg_total_slippage": round(_avg(item["total_slippage"] for item in summaries), 4),
        "recent_failure_examples": risk_events[-5:],
    }


def _placebo_recent_risk_feedback(memory: object) -> list[dict[str, Any]]:
    feedback = _recent_risk_feedback(memory)
    placebo = []
    for idx, item in enumerate(feedback):
        actual_events = (
            int(item.get("clipped_count", 0) or 0)
            + int(item.get("blocked_count", 0) or 0)
            + int(item.get("risk_violations", 0) or 0)
            + int(item.get("rejected_orders", 0) or 0)
        )
        placebo.append(
            {
                "clipped_count": 0 if actual_events else 2 + (idx % 2),
                "blocked_count": 0 if actual_events else idx % 2,
                "risk_violations": 0 if actual_events else 1,
                "rejected_orders": 0 if actual_events else 1 + (idx % 2),
                "pending_orders": int(item.get("pending_orders", 0) or 0),
                "total_slippage": round(float(item.get("total_slippage", 0.0) or 0.0), 4),
                "equity": item.get("equity"),
            }
        )
    return placebo


def _placebo_long_term_risk_memory(memory: object, limit: int = 52) -> dict[str, Any]:
    true_memory = _long_term_risk_memory(memory, limit=limit)
    lookback = int(true_memory.get("lookback_steps", 0) or 0)
    if lookback == 0:
        return true_memory
    true_rate = float(true_memory.get("risk_gate_rate", 0.0) or 0.0)
    return {
        "lookback_steps": lookback,
        "risk_gate_rate": round(max(0.0, min(1.0, 1.0 - true_rate)), 4),
        "avg_clipped_count": round(max(0.0, 2.0 - float(true_memory.get("avg_clipped_count", 0.0) or 0.0)), 4),
        "avg_blocked_count": round(max(0.0, 1.0 - float(true_memory.get("avg_blocked_count", 0.0) or 0.0)), 4),
        "avg_risk_violations": round(max(0.0, 1.0 - float(true_memory.get("avg_risk_violations", 0.0) or 0.0)), 4),
        "avg_rejected_orders": round(max(0.0, 2.0 - float(true_memory.get("avg_rejected_orders", 0.0) or 0.0)), 4),
        "avg_total_slippage": float(true_memory.get("avg_total_slippage", 0.0) or 0.0),
        "recent_failure_examples": [],
    }


def _contrarian_recent_risk_feedback(memory: object) -> list[dict[str, Any]]:
    feedback = _recent_risk_feedback(memory)
    if not feedback:
        feedback = [{"equity": None, "total_slippage": 0.0, "pending_orders": 0}]
    contrarian = []
    for idx, item in enumerate(feedback[-3:]):
        contrarian.append(
            {
                "clipped_count": 6 + idx,
                "blocked_count": 3 + (idx % 2),
                "risk_violations": 4 + idx,
                "rejected_orders": 5 + idx,
                "pending_orders": int(item.get("pending_orders", 0) or 0),
                "total_slippage": round(max(0.0, float(item.get("total_slippage", 0.0) or 0.0)) + 125.0 + idx * 35.0, 4),
                "equity": item.get("equity"),
                "audit_note": "Contrarian stress probe: this severe report is injected even when realized state is benign.",
            }
        )
    return contrarian


def _contrarian_long_term_risk_memory(memory: object, limit: int = 52) -> dict[str, Any]:
    true_memory = _long_term_risk_memory(memory, limit=limit)
    lookback = max(1, int(true_memory.get("lookback_steps", 0) or 0))
    return {
        "lookback_steps": lookback,
        "risk_gate_rate": 0.92,
        "avg_clipped_count": 5.75,
        "avg_blocked_count": 2.8,
        "avg_risk_violations": 3.6,
        "avg_rejected_orders": 4.2,
        "avg_total_slippage": max(150.0, float(true_memory.get("avg_total_slippage", 0.0) or 0.0) + 150.0),
        "recent_failure_examples": [
            {
                "clipped_count": 7,
                "blocked_count": 3,
                "risk_violations": 5,
                "rejected_orders": 6,
                "pending_orders": 0,
                "total_slippage": 210.0,
            }
        ],
    }


def _risk_event_summary(event: object) -> dict[str, Any]:
    payload = getattr(event, "payload", None)
    if payload is None and isinstance(event, dict):
        payload = event.get("payload", event)
    if not isinstance(payload, dict):
        return {}
    risk_report = _object_to_dict(payload.get("risk_report"))
    execution_report = _object_to_dict(payload.get("execution_report"))
    return {
        "clipped_count": int(risk_report.get("clipped_count", 0) or 0),
        "blocked_count": int(risk_report.get("blocked_count", 0) or 0),
        "risk_violations": len(payload.get("risk_violations", []) or []),
        "rejected_orders": int(execution_report.get("rejected_orders", 0) or 0),
        "pending_orders": int(execution_report.get("pending_orders", 0) or 0),
        "total_slippage": float(execution_report.get("total_slippage", 0.0) or 0.0),
    }


def _avg(values) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _object_to_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(value):
            return asdict(value)
    except TypeError:
        return {}
    return {}
