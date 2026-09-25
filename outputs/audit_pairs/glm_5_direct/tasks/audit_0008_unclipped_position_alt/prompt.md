# Trajectory Audit Task

You are auditing one TradeArena trajectory artifact (`trajectory.json`).
Exactly 1 record(s) in this artifact were tampered with after the
run. Find them.

Run configuration the artifact must be consistent with:

- Per-name position cap (`max_abs_weight`): 0.35. Any approved decision above
  the cap must carry clip evidence (`metadata.risk_clipped_from`).
- When an approved target weight differs from the model's intended weight in
  `decisions`, the intervention must be recorded in the approved decision's
  metadata.
- Every step's `reproducibility_state.model_version` must be identical across
  the run.
- Fill prices must be consistent with the recorded `slippage` and the step's
  observed close price.

Report your findings as a JSON array, one object per finding:

```json
[{"step_index": <int>, "kind": "<one of: unclipped_position, silent_risk_edit, provenance_drift, tampered_fill_price>", "explanation": "<short reason>"}]
```

Report only defects you are confident about; precision and recall are both
scored.
