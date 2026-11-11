# CS781 methodology documentation overlay (local, not pushed)

**Public repo target:** https://github.com/k27321278-a11y/cs781-capstone-code

**This is an additive documentation overlay, not a replacement for the code or for the official course report.** It contains 61-variant PDF coverage audit, four source-derived prepared Phase 3 methodology reports (no scores claimed), and one expanded source-derived Phase 5 sparse-hybrid methodology supplement. Files are under matching repository-relative paths so you can place them into your clone without overwriting existing model code or scores.

**GitHub upload status:** NOT PUBLISHED. The connected GitHub app returned HTTP 403 (`Resource not accessible by integration`) when trying to create a documentation audit file on `main`. The live repo itself remains public. The user or a sufficiently authorized GitHub connection must commit/push this overlay.

## Add via a normal Git clone

1. Clone the public CS781 repo locally (or open your existing clone).
2. Extract this overlay and copy the contents of `github_overlay/` into the repository root, preserving directories.
3. Review the new documentation, then use `git status` to ensure only new documentation files appear; no original source should be modified.
4. Run `git add docs/PHASE1_TO_7_METHODOLOGY_COVERAGE_AUDIT_20261112* phase3/methodology phase5/variants/b1_sparse_hybrid/report/A0_DEPTH_METHODS_SUPPLEMENT.* phase5/variants/b1_sparse_hybrid/diagrams/` followed by your own descriptive `git commit` and `git push origin main`.
5. Confirm the resulting links load when signed out of GitHub. Do not add original checkpoint weights or official large datasets unless the course requires them.

## Status / integrity

- Four Phase 3 descriptions are for **prepared, unscored** methods; no metrics fabricated.
- Expanded Phase 5 report includes actual source line references and archived scores; no new training run.
- Source files in the old repository and course ZIP were **not** rewritten.
- The audit explicitly marks **56 further existing short/unverified-depth reports** for source-by-source expansion. A0-depth completeness of all 61 is **not** claimed.
