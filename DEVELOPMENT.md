# Development checks

The primary reader command is `python -B scripts/verify_tmlr_supplement.py`.
That command checks every required input and reproduces all 15 registered
tables, appendix, and figure outputs without network access. Its historical
filename does not imply that the paper is submitted to TMLR. The manuscript is
not included, so `scripts/verify_report_claims.py` runs its data-side checks
and reports that the paper-side half was skipped; every data input is still
required.

For the regression tests, install pytest in addition to
`requirements-figures.txt`, then run:

```text
python -B -m pytest tests/
```

The tests disable pytest's cache and `-B` avoids adding bytecode to the frozen
tree. The verifier permits only root Git administration files outside the
manifest; unexpected files still cause a failure. If developing with mypy,
Ruff, or compileall, direct their caches outside this tree before validating
the exact artifact file set.

`artifact_manifest.json` binds the released bytes. It is an integrity check,
not a third-party timestamp or proof of provider authenticity. Do not overwrite
the frozen plan or raw responses to make a changed analysis pass. A revised
release should explain any changed evidence or analysis and regenerate its
manifest openly.
