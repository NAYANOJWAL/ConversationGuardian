# Antigravity Work Order 001 — Read-Only Integrity Audit

**Priority:** P0  
**Mode:** Audit only; no code or data edits  
**Branch:** `work/technical-direction`  
**Decision owner:** ChatGPT  
**Worker:** Antigravity  
**User:** final approver

## Required reading

Read these first:
- `docs/TECHNICAL_DIRECTION_AND_NOVELTY.md`
- `README.md`
- `docs/RESULTS.md`
- `docs/NOSQL_DESIGN.md`
- `docs/ASSESSMENT_GUIDE.md`
- `scripts/run_sequence_benchmark.py`
- `app/database.py`
- `app/dashboard.py`
- `tests/` and `.github/workflows/checks.yml`

## Hard stop conditions

- Do not edit, format, move, delete, or create source/data/test/report artifacts during this audit.
- Do not change the baseline branch or the user's original local `ConversationGuardian_Final`.
- Do not run training or a full benchmark that overwrites current generated artifacts.
- Do not print any token, password, URL containing credentials, or secret value.
- Do not assume GitHub access proves the local workspace contains the same commit.
- Report missing access and uncertainty explicitly.

## Audit tasks

1. Record branch, HEAD SHA, clean/dirty status, tracked file tree, remotes, and local-vs-remote divergence without changing the repository.
2. Check the exact commit from which `work/technical-direction` was created.
3. Compare the current README evaluation description with `docs/RESULTS.md`, benchmark implementation, CI workflow, and tests. Explicitly distinguish:
   - current 2,508-row snapshot, 80/20 internal holdout, seed 42;
   - historical 4,188-conversation split results;
   - original DistilBERT full-thread evaluation;
   - dashboard live simulator results.
4. For every model result shown in the README, docs, dashboard, and figures, identify its true input, source artifact, split, threshold, and score type. Record any inconsistency.
5. Verify all benchmark rows, prefixes/horizons, and output filenames expected by tests and CI. Check for stale output artifacts or unreachable paths.
6. Review prefix handling carefully: whether 100% for an attack-labelled conversation can contain the annotated attack turn, whether the benchmark's current snapshot is already pre-truncated, and whether the code can honestly call all tested prefixes “pre-attack.”
7. Review the live Risk Simulator's use of the DistilBERT checkpoint. Identify whether it is full-thread scoring, prefix forecasting, or both. Do not infer calibration from softmax output.
8. Verify NoSQL database mode detection, idempotent seeding, moderator note CRUD, reference data protection, cache key/TTL, cache counters, service error behavior, and whether there is any actual review-event/history model today.
9. Inspect NoSQL benchmark timed functions and raw outputs. State exactly what work each timer measures; distinguish `mongomock`/`fakeredis` simulations from live MongoDB/Redis.
10. Inspect index/explain logic. Do not claim actual COLLSCAN vs IXSCAN comparison unless saved live server output exists.
11. Check for secret-handling risks, including notebook history and repository tracking. Report only file and line/cell locations and whether rotation is indicated; never print a secret.
12. Evaluate the current proposed novelty against existing functionality. Separate what exists today from what is proposed.
13. Produce an audit report with a table: severity, finding, evidence path/line, impact on paper/demo, exact smallest remediation. Do not implement fixes.
14. Return a concise handoff containing command outputs summarized, all unresolved assumptions, and the recommended next gate. Stop for ChatGPT review.

## Definition of done

A factual, evidence-linked report exists, with a clear PASS / FAIL / UNKNOWN for each item. No code, data, test, generated artifact, branch, or user baseline has been changed. No claims of completion without exact evidence.
