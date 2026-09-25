# Measurement Boundaries in LLM Financial Agent Evaluation

Fixed-Tape Execution and Multi-Defect Auditing

**Paper:** author-signed preprint, distributed separately; an arXiv identifier
will be added here after submission.

**Repository:** https://github.com/weich97/llm-evaluation-boundaries-artifact

This repository accompanies the author-signed preprint. Study A distinguishes a
fixed-response mechanical execution contrast from an adaptive-agent comparison.
Study B checks target recall beside false positives and exact-set correctness
in matched zero-, one-, and two-defect tasks. The studies do not establish a
common model failure mechanism, real-market trading ability, or reliable
end-to-end oversight. Study C belongs to a separate project and is not included.

The manuscript is not part of this repository. `paper/tech_report/` holds only
the generated appendix fragment and the three figures that the verifier
regenerates and compares byte for byte against the released reference files.
No arXiv posting has been performed by preparing this repository.

## Revision 1.1.1 (2026-09-24)

The appendix uses the manuscript's estimand labels. The Study B caption now
distinguishes unadjusted 95% confidence intervals from Holm-corrected test
p-values. The citation title matches the manuscript's punctuation. These are
reporting updates: the frozen inputs, analysis results and figures are unchanged.

## Offline reproduction

Use Python 3.11 or newer. Install the dependencies in `requirements-figures.txt`
beforehand, then disconnect from the network and run from the extracted root:

```text
python -B scripts/verify_tmlr_supplement.py
```

The verifier uses a temporary copy. It checks the exact file manifest (excluding
only the checkout's root `.git` administration directory), complete
task/result grids and rule oracles, reconstructs hashed prompts, reparses all
600 primary responses, regenerates tables and three PDF figures, and compares
outputs with the reference files. Required checks cannot skip. Analysis workers
reject network connections. No credentials or model access are needed.
Frozen tasks, decoding settings and the six-test primary family are unchanged.

## Coverage and limits

| Claim family | Starting evidence | Recomputed here | Not reconstructed |
|---|---|---|---|
| A fixed-tape contrasts and ranking | 1,800 replay outcomes, 900 path records, frozen plan/integrity records | Cluster intervals, decomposition, rankings, divergence, appendix and figure | Original provider calls and simulator trajectories |
| B primary matched triplets | 300 tasks and keys, 60 trading source artifacts, 600 results and exact response texts | Task rules, request hashes, strict parsing including nine failures, scoring, McNemar/Holm, bootstrap tables | Service authenticity or fresh repeatability |
| B legacy answer-key correction | Original keys, structured findings and task prompts | Target-label re-scoring, quantity-cue counterexample and figure | Parsing withheld legacy response text |
| B interventions and self-audit | Original keys and complete structured findings | Target hits, three-sample votes, matched tests, Fisher tests and approximate MDE | Parsing withheld secondary response text |

`evidence/primary_responses.jsonl` preserves exact response text, including
malformed output. Prompts are reconstructed from the frozen tasks. Answer keys
are supplied for offline review and are not auditor inputs. Legacy hit flags
are recomputed from findings before analysis. The original legacy keys are
intentionally retained: their omission is being corrected, not silently hidden.

`coverage_decomposition.csv` is a descriptive post-hoc breakdown, not a new
primary test. Internal digests establish consistency, not independent collection
timestamps, provider authenticity or preregistration. Source-matrix provider
metadata are minimal projections for the grid/temperature checks, not full
private provenance. Code is covered by the included MIT license.

## Figure sources

| Figure | Script | Inputs |
|---|---|---|
| execution_identification.pdf | scripts/render_studyA_figures.py | released replay rows and derived A tables |
| audit_cardinality.pdf | scripts/render_finaudit_figures.py | legacy findings, keys and corrected-key table |
| pooling_hides_structure.pdf | scripts/render_pooling_figures.py | path divergence and matched-triplet tables |

PDF metadata timestamps are suppressed. With the pinned rendering stack the
verifier compares generated PDF bytes, as well as tables and the appendix.
All three renderers use `scripts/report_figure_style.py`: STIX serif text,
9-point annotations and ticks, 10-point axis labels, and 11-point panel titles.
Each PDF is exported at the manuscript's 6.5-inch text width, so those sizes are
preserved when included at full width. The PDF fonts are embedded TrueType.
The substrate paper is cited in the preprint; its text, figures and results are
not bundled. No general-purpose model cache, credential, private key or
scheduler state is included.

## Release scope and attribution

Only selected frozen inputs under `outputs/` are included. They are necessary
for offline reconstruction, not a copy of the development cache. The manuscript
explains which raw responses are released and which secondary analyses start
from structured findings. Provider model identifiers are not immutable weights.
The original MIT license and upstream copyright notice are retained for the
software. The preprint license and any arXiv license selection remain separate
from the software license.
