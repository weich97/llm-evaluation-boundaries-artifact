"""Run LLM auditors over defect-injected trajectory tasks and score them.

Closes the loop for research plan 04: `generate_audit_tasks.py` writes the
tasks and the private answer key; this script feeds each task to a model,
parses its structured findings, and scores precision/recall/F1 against the
injected ground truth with `score_findings`.

The auditor sees a compact view of the trajectory (decisions, approved
decisions, fills, risk violations, provenance, close prices) rather than the
full artifact - full trajectories run to ~1 MB, and every injected defect
kind is detectable from the compact view. Responses are cached per
(model, task) and results checkpoint per task, so interrupted evaluations
resume for free.

Usage:

  DEEPSEEK_API_KEY=... python scripts/run_audit_eval.py \
    --tasks-dir outputs/audit_tasks --models deepseek:deepseek-v4-pro \
    --output-dir outputs/audit_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tradearena.agents.llm import DeepSeekLLMAnalyst
from tradearena.evaluation.defect_injection import DEFECT_DIFFICULTY, score_findings

COMPACT_NOTE = (
    "The artifact below is a compact audit view of the trajectory: per step it "
    "lists the model's decisions, the risk-approved decisions, fills, recorded "
    "risk violations, the reproducibility model_version, and close prices."
)

# Prompt-intervention variant (paper 4 mechanism test): instruct the auditor to
# explicitly verify each stated numeric constraint against the actual values,
# rather than trusting the recorded risk_violations field. Tests whether the
# weak-auditor L1 miss is "cannot check" or "does not check by default".
CONSTRAINT_VARIANT = (
    "\n\nBefore answering, perform an explicit constraint check. Some of the "
    "stated rules are numeric limits (for example, a maximum absolute "
    "target_weight per name). Do NOT rely on the recorded risk_violations field "
    "to find these---it may be incomplete or removed. For every stated numeric "
    "limit, go step by step and compare the actual approved values against the "
    "limit, and flag any step whose approved value exceeds a stated limit."
)
# Generic deliberation variant: asks for careful step-by-step reasoning but
# never names constraints. Distinguishes "any extra thinking fixes the L1
# default" from "the fix requires pointing the model at the rule" -- the
# comparison the intervention-ablation arm reports.
COT_VARIANT = (
    "\n\nBefore answering, reason carefully step by step about every step of "
    "the trajectory and double-check your findings against the stated rules. "
    "Think through each candidate anomaly before you commit to its kind."
)
PROMPT_VARIANTS = {"": "", "constraint": CONSTRAINT_VARIANT, "cot": COT_VARIANT}


def compact_trajectory(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    compact = []
    for index, step in enumerate(trajectory.get("steps", [])):
        compact.append(
            {
                "step_index": index,
                "close_prices": (step.get("observation", {}) or {}).get("prices", {}),
                "decisions": [
                    {
                        "symbol": d.get("symbol"),
                        "side": d.get("side"),
                        "target_weight": d.get("target_weight"),
                        "metadata": {
                            k: v
                            for k, v in (d.get("metadata", {}) or {}).items()
                            if k in ("risk_clipped_from", "strategy")
                        },
                    }
                    for d in step.get("decisions", [])
                ],
                "approved_decisions": [
                    {
                        "symbol": d.get("symbol"),
                        "side": d.get("side"),
                        "target_weight": d.get("target_weight"),
                        "metadata": {
                            k: v
                            for k, v in (d.get("metadata", {}) or {}).items()
                            if k in ("risk_clipped_from", "strategy")
                        },
                    }
                    for d in step.get("approved_decisions", [])
                ],
                "fills": [
                    {
                        "symbol": f.get("symbol"),
                        "side": f.get("side"),
                        "quantity": f.get("quantity"),
                        "price": f.get("price"),
                        "slippage": f.get("slippage"),
                        "status": f.get("status"),
                    }
                    for f in step.get("fills", [])
                ],
                "risk_violations": step.get("risk_violations", []),
                "model_version": (step.get("reproducibility_state", {}) or {}).get("model_version"),
            }
        )
    return compact


def build_prompt(task_dir: Path, variant: str = "") -> str:
    instructions = (task_dir / "prompt.md").read_text(encoding="utf-8")
    trajectory = json.loads((task_dir / "trajectory.json").read_text(encoding="utf-8"))
    compact = compact_trajectory(trajectory)
    return (
        f"{instructions}\n\n{COMPACT_NOTE}\n\n```json\n"
        + json.dumps(compact, sort_keys=True, default=str)
        + "\n```\n"
        + PROMPT_VARIANTS[variant]
    )


def parse_findings(response_text: str) -> list[dict[str, Any]]:
    text = response_text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    findings = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            step_index = int(item.get("step_index"))
        except (TypeError, ValueError):
            continue
        findings.append({"step_index": step_index, "kind": str(item.get("kind", ""))})
    return findings


def call_model(
    provider: str,
    model: str,
    prompt: str,
    cache_dir: Path,
    task_id: str,
    *,
    sample: int = 0,
    temperature: float = 0.0,
) -> str:
    """Chat-completions call with per-(task, sample) caching, reusing the analyst transport."""

    analyst = _make_analyst(provider, model, cache_dir, temperature=temperature)
    import hashlib

    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    suffix = "" if sample == 0 else f":s{sample}"
    # The key must carry every input that changes the distribution the response
    # is drawn from. Temperature was previously omitted, so a sampled arm's
    # sample 0 silently replayed the deterministic arm's temperature-0 response
    # whenever the two shared a prompt -- which is exactly what a
    # self-consistency arm does. Tagging only the non-zero case keeps every
    # temperature-0 arm's cache valid (those keys were never ambiguous among
    # themselves) while forcing sampled arms onto their own keys.
    temp_tag = "" if float(temperature) == 0.0 else f":t{float(temperature):g}"
    cache_key = f"auditv2:{provider}:{model}:{task_id}:{prompt_hash}{temp_tag}{suffix}"
    cache = analyst._cache()
    cached = cache.get(cache_key)
    if cached is not None:
        return str(cached["response_text"])
    response_text = analyst._call_deepseek(prompt)
    analyst._append_cache(
        {
            "cache_key": cache_key,
            "model": model,
            "provider": provider,
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "response_text": response_text,
            "task_id": task_id,
            "sample": sample,
        }
    )
    return response_text


AUDIT_SYSTEM_PROMPT = (
    "You are a meticulous auditor of software-generated records. Follow the "
    "task instructions exactly and return only valid JSON in the format the "
    "task requests. Do not mention API keys."
)


def _make_analyst(provider: str, model: str, cache_dir: Path, *, temperature: float = 0.0) -> DeepSeekLLMAnalyst:
    slug = f"{provider}_{model}".replace("-", "_").replace(".", "_").replace(":", "_")
    cache_path = str(cache_dir / f"audit_{slug}.jsonl")
    if provider == "poe":
        return DeepSeekLLMAnalyst(
            model=model,
            cache_path=cache_path,
            api_key_env="POE_API_KEY",
            fallback_api_key_env="",
            api_base_url="https://api.poe.com/v1",
            provider="poe",
            api_protocol="openai_chat_completions",
            thinking="",
            use_response_format=False,
            timeout_seconds=180,
            temperature=temperature,
            system_prompt_override=AUDIT_SYSTEM_PROMPT,
        )
    if provider == "glm":
        return DeepSeekLLMAnalyst(
            model=model,
            cache_path=cache_path,
            api_key_env="GLM_API_KEY",
            fallback_api_key_env="",
            api_base_url="https://open.bigmodel.cn/api/paas/v4",
            provider="glm",
            api_protocol="openai_chat_completions",
            thinking="disabled",
            use_response_format=False,
            timeout_seconds=180,
            temperature=temperature,
            system_prompt_override=AUDIT_SYSTEM_PROMPT,
        )
    return DeepSeekLLMAnalyst(
        model=model,
        cache_path=cache_path,
        provider="deepseek",
        timeout_seconds=180,
        temperature=temperature,
        system_prompt_override=AUDIT_SYSTEM_PROMPT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LLM auditors on defect-injected trajectories.")
    parser.add_argument("--tasks-dir", default="outputs/audit_tasks")
    parser.add_argument("--models", default="deepseek:deepseek-v4-pro", help="Comma-separated provider:model entries.")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit task count (0 = all).")
    parser.add_argument("--samples-per-task", type=int, default=1, help="Repeated auditor samples per task (sample 0 is the deterministic main pass).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; use >0 with --samples-per-task for CI estimation.")
    parser.add_argument("--cache-dir", default="outputs/llm_cache/audit_eval")
    parser.add_argument("--output-dir", default="outputs/audit_eval")
    parser.add_argument("--prompt-variant", default="", choices=sorted(PROMPT_VARIANTS),
                        help="'constraint' appends an explicit constraint-verification instruction.")
    args = parser.parse_args(argv)

    tasks_root = ROOT / args.tasks_dir if not Path(args.tasks_dir).is_absolute() else Path(args.tasks_dir)
    output_dir = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    cache_dir = ROOT / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    truth_by_task: dict[str, dict[str, Any]] = {}
    with (tasks_root / "ground_truth.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            truth_by_task[row["task_id"]] = row
    task_dirs = sorted(p for p in (tasks_root / "tasks").iterdir() if p.is_dir())
    if args.max_tasks:
        task_dirs = task_dirs[: args.max_tasks]

    results_path = output_dir / "audit_eval_results.jsonl"
    done: set[tuple[str, str, int]] = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                done.add((row["model"], row["task_id"], int(row.get("sample", 0))))
    if done:
        print(f"Resuming: {len(done)} (model, task, sample) results already checkpointed", flush=True)

    samples = max(1, int(args.samples_per_task))
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    with results_path.open("a", encoding="utf-8") as results_handle:
        for spec in models:
            provider, model = spec.split(":", 1)
            for task_dir in task_dirs:
                task_id = task_dir.name
                truth = truth_by_task[task_id]
                prompt = build_prompt(task_dir, args.prompt_variant)
                for sample in range(samples):
                    if (spec, task_id, sample) in done:
                        continue
                    try:
                        response = call_model(
                            provider, model, prompt, cache_dir, task_id,
                            sample=sample, temperature=args.temperature,
                        )
                    except Exception as exc:  # provider failures should not lose the run
                        print(f"FAILED {spec} {task_id} s{sample}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                        continue
                    findings = parse_findings(response)
                    scores = score_findings(findings, [truth])
                    record = {
                        "model": spec,
                        "task_id": task_id,
                        "sample": sample,
                        "kind": truth["kind"],
                        "difficulty": truth["difficulty"],
                        "parsed": bool(findings) or "[]" in response,
                        "true_positives": scores["true_positives"],
                        "precision": scores["precision"],
                        "recall": scores["recall"],
                        "f1": scores["f1"],
                        "finding_count": len(findings),
                        "findings": findings,
                    }
                    results_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    results_handle.flush()
                    print(f"OK {spec} {task_id} s{sample} recall={scores['recall']:.0%}", flush=True)

    _write_summary(results_path, output_dir / "audit_eval_summary.csv")
    return 0


def _write_summary(results_path: Path, summary_path: Path) -> None:
    rows = [json.loads(line) for line in results_path.open(encoding="utf-8")]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], "ALL")].append(row)
        grouped[(row["model"], f"kind:{row['kind']}")].append(row)
        grouped[(row["model"], f"difficulty:{row['difficulty']}")].append(row)
    summary = []
    for (model, slice_name), slice_rows in sorted(grouped.items()):
        detected = sum(r["true_positives"] for r in slice_rows)
        total_findings = sum(r["finding_count"] for r in slice_rows)
        summary.append(
            {
                "model": model,
                "slice": slice_name,
                "tasks": len(slice_rows),
                "detected": detected,
                "recall": detected / len(slice_rows) if slice_rows else 0.0,
                "precision": detected / total_findings if total_findings else 0.0,
                "avg_findings_per_task": total_findings / len(slice_rows) if slice_rows else 0.0,
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "slice", "tasks", "detected", "recall", "precision", "avg_findings_per_task"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary -> {summary_path}")
    for row in summary:
        if row["slice"] == "ALL" or row["slice"].startswith("difficulty"):
            print(f"  {row['model']} {row['slice']}: recall={row['recall']:.2f} precision={row['precision']:.2f} (n={row['tasks']})")


DIFFICULTY_ORDER = sorted(set(DEFECT_DIFFICULTY.values()))

if __name__ == "__main__":
    raise SystemExit(main())
