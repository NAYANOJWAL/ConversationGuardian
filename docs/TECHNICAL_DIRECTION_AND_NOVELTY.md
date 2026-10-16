# ConversationGuardian — Technical Direction, Novelty, and Validation Gates

**Decision owner:** ChatGPT (technical architecture and acceptance decisions)  
**Implementation worker:** Antigravity  
**Project operator and final approver:** User  
**Baseline branch:** `upgrade/nosql-presentation`  
**Working branch:** `work/technical-direction`

## 1. Executive decision

Keep ConversationGuardian as one connected system for human-in-the-loop moderation support. Do not add a complex neural architecture simply to claim novelty. The present evidence does not establish that ordered state n-grams alone significantly outperform orderless state features. The modest, defensible contribution should be an **auditable, sequentially informed moderation worklist** that connects (a) what context was available when a risk was estimated, (b) the risk estimate and evidence behind it, (c) moderator review actions, and (d) the data access/caching behavior needed to support that workflow.

Suggested working title:

**ConversationGuardian: An Auditable Sequential Decision-Support and NoSQL Workbench for Early Conversation-Escalation Review**

This is applied-system novelty, not a claim of a new classifier or a first-of-its-kind algorithm. The report must compare with relevant prior systems and state that contribution narrowly.

## 2. Problem statement

Moderators of online discussion threads need more than a full-thread toxicity score. They need to know what the system could observe at a chronological point in a conversation, what signal it produced, what evidence it used, and whether a human reviewer followed up. ConversationGuardian should support this workflow while preserving a clear distinction between early-prefix forecasts and post-hoc detection.

**Research question:** How can leakage-aware prefix-level conversation-risk estimates be connected to an auditable moderator-review workflow, and what practical role do document-oriented storage, indexes, and score caching play in retrieving evidence and supporting repeat analyses?

### Scope boundaries

- Target label: whether a conversation is attack-labelled overall in the Conversations Gone Awry Wikipedia corpus.
- Early warning: predictions from 25%, 50%, and 75% chronological prefixes; 100% is a full-context reference, not an early-warning claim.
- Intended user: human moderator or researcher investigating discussion risk.
- System output: review-priority evidence, not an automatic enforcement decision.
- Current benchmark boundary: the public repository's current `docs/RESULTS.md` describes a stratified 80/20 internal holdout drawn from the bundled 2,508-conversation processed training snapshot (2,006 train / 502 holdout, seed 42). Do not conflate this with the historic 4,188-conversation partition experiment or call it the official CGA-Wiki test benchmark.

## 3. Small novelty: Auditable Prefix-to-Review Trace

Implement this as the main addition. It connects the current modeling and database components rather than replacing them.

For each analysis or saved review event, preserve a structured trace with:
- stable event ID and conversation ID;
- timestamp and model/preprocessing version;
- horizon / number or fraction of observed turns;
- risk score and the score type (e.g. DistilBERT softmax vs. benchmark model probability);
- message-level signals and state trajectory available at that horizon;
- matching pattern ID and support counts, with an evidence-strength flag;
- recommendation shown to the user, not an enforcement action;
- moderator note/status and subsequent review timestamp, if recorded;
- whether the event was computed with persistent MongoDB/Redis or in-memory demo modes.

Do not store a future attack turn or future-derived signal in a prefix event. Do not treat pattern support ratios as causal risk multipliers. Do not silently mix the current dashboard's full-thread DistilBERT score with the separate multi-horizon TF-IDF/state benchmark. If the live simulator cannot produce validated prefix-level forecasts from its current checkpoint, label it clearly as a full-thread demonstration until that bridge is implemented and tested.

### Why this is useful

A moderator can answer: “What did the system know at 25%, 50%, or 75% of the discussion, what triggered review, and what did the reviewer do afterward?” A researcher can audit whether warning estimates change with context, and a database examiner can inspect the stored trace and review state. This is the integration contribution; model accuracy gains are a separate empirical question.

## 4. Sequential modeling: what it must prove

Retain the existing leakage-aware multi-horizon benchmark and baseline comparisons. Do not assert that order itself provides a statistically significant benefit unless the evidence supports it.

### Required experiments

1. Use exactly one documented data snapshot and one explicit split protocol per reported table. Save IDs/hashes and split metadata.
2. Evaluate 25%, 50%, and 75% early horizons; show 100% separately as full-context reference.
3. Compare majority reference, message-count/length-only baseline, word TF-IDF, word+character TF-IDF, orderless states, ordered state n-grams, and hybrid text+state.
4. Use validation data for threshold selection; never tune thresholds on the held-out evaluation subset.
5. Keep conversation ID as the split unit and test that there is no train/holdout overlap.
6. Report Macro-F1, attack precision/recall/F1, PR-AUC, ROC-AUC, balanced accuracy, Brier score, confusion matrix, and uncertainty intervals when the experiment can calculate them reliably.
7. Run paired bootstrap comparisons for hybrid vs text-only, hybrid vs orderless state, and ordered vs orderless state, over identical held-out conversations. State clearly which intervals include zero.
8. Audit conversation-length confounding. Do not confuse shorter pre-attack eligible context with the act of being attack-labelled.
9. Preserve predictions and configuration for reproducibility.
10. State historic test-set use transparently. External/generalization claims require a truly separate evaluation source or a newly created untouched holdout protocol.

### Interpretation rule

If the hybrid is not consistently better, do not manufacture a model claim. The accepted contribution can be that a moderate, transparent forecasting signal is integrated into a useful review/audit workflow. A negative or mixed modeling result is acceptable when reported honestly.

## 5. NoSQL: make the technology operationally necessary

The storage layer should retain distinct responsibilities:

- **MongoDB `conversations`:** read-heavy reference documents, with utterances embedded because the main workflow inspects a thread and its message signals together.
- **MongoDB `mined_patterns`:** descriptive pattern catalog with support counts and reliability warnings.
- **MongoDB `review_events` (proposed):** append-oriented record of prefix-level assessments; records should include model version, horizon, timestamp, and evidence references. If the implementation instead extends another collection, justify it and preserve the ability to audit each event.
- **MongoDB `moderator_notes`:** mutable review notes kept separate from source conversation text.
- **Redis:** cache repeated per-message toxicity score computations using a SHA-256 text digest and TTL. Cache only the score/metadata needed; don't store raw text as the key.

### Required query workloads

1. Retrieve a conversation and its chronological signals.
2. Find saved review events for a conversation and order them by time/horizon.
3. Retrieve unresolved moderation notes or reviews for a worklist.
4. Search nested speakers and run aggregation of high-toxicity comments by speaker.
5. Read/update/delete moderator notes while leaving reference corpus untouched.
6. Repeat a message-analysis request and observe cache hit/miss behavior.

### Benchmark requirements

- Unit tests with mocks establish functional behavior only.
- For real MongoDB and Redis claims, run actual services and record server/version/configuration. Verify persistence across app restart.
- Test query plans with `explain("executionStats")`, save winning plan, documents examined, keys examined, and returned count. Do not claim a speedup unless measured.
- Benchmark Redis with an actual live Redis connection and a real, consistent cache miss computation; compare uncached model inference with cache hit using the same scoring function, and include warm-up, sample count, median and tail percentiles. A `time.sleep` proxy and `fakeredis` are simulation only.
- Report when an index does not improve performance. Tiny datasets and mock implementations may not show realistic database query behavior.
- Test cache invalidation or TTL behavior and document what happens when a service is unavailable.
- No credentials, access tokens, model secrets, or `.env` values may be committed. Rotate any real token ever exposed in notebook history.

## 6. Integrated dashboard acceptance criteria

The four current tabs may remain: Risk Simulator, Pattern Catalog, NoSQL Lab, Sequential Results. Add a concise **Review Trace / Worklist** view or integrate it where it fits best.

A successful demonstration should show one journey end-to-end:
1. Select a labelled demo scenario or a clearly disclosed sample.
2. Display turn-level sentiment/toxicity and state sequence.
3. Show the chosen prefix horizon and its available evidence; prevent future turns from influencing the prefix result.
4. Display model score, score type, model version, and limitations.
5. Save a review event and moderator note.
6. Retrieve the event/history from MongoDB and show stable IDs and timestamps.
7. Re-run identical message analysis and demonstrate cache hit/miss using actual service or clearly marked demo mode.
8. Run one nested-field query and aggregation.
9. Show a real query execution plan only when connected to MongoDB.
10. Edit and delete a note; verify the reference corpus remains unchanged.
11. If claiming persistence, restart/refresh and prove saved state survives against real MongoDB.
12. Show an honest status badge/caption identifying live services versus mocks; do not present a mock as a live database.

## 7. Report strategy

Prepare a complete IEEE conference-style report only from reproducible artifacts. Include at least 15 directly relevant, verified related works and a comparative literature table. Report the system contribution accurately as an integrated workflow unless further evidence establishes more.

Recommended structure:
1. Abstract and keywords
2. Introduction and motivation
3. Related work (minimum 15 research studies; separate official docs from research papers)
4. Problem definition and research questions
5. Contributions
6. Dataset and leakage-aware protocol
7. Sequential methods and baselines
8. NoSQL schema, indexes, cache, and workload
9. Auditable prefix-to-review workflow and dashboard
10. Experimental setup and software verification
11. Sequential results (tables, horizon charts, PR curves/confusion matrices as supported)
12. NoSQL results (query plans and latency distributions only when live results exist; mocks clearly labelled)
13. Integrated end-to-end demonstration and case walkthrough
14. Limitations, ethics/fairness, and threats to validity
15. Conclusion and future work
16. References and reproducibility appendix

Each plot/table should be generated from saved CSV/JSON or raw experiment output. Include titles, units, sample counts, confidence intervals where supported, and a nearby interpretation. Don't create numbers manually to fill a table. Mark incomplete items as `[PENDING: exact evidence needed]`.

## 8. Live demo plan (7–9 minutes)

- 0:00–0:45: Problem, user, and non-enforcement scope.
- 0:45–2:00: Sequential Results: compare the same three horizons for text-only, orderless, ordered, and hybrid approaches. Explain that full context is a reference rather than early warning.
- 2:00–3:15: Run one healthy discussion and one tense/escalation scenario. Inspect turn signals, state trajectory, and score type. Avoid claiming the synthetic preset is a labelled test example.
- 3:15–4:30: Review Trace / worklist: show the saved event, horizon, timestamp, model version, evidence, and reviewer action.
- 4:30–6:15: NoSQL Lab: show document shape, nested speaker query, aggregation, and explain plan when real MongoDB is available.
- 6:15–7:15: Save/read/update/delete a moderator note, then restart/refresh to show persistence if using live services.
- 7:15–8:00: Repeat message scoring to demonstrate cache hit/miss using actual Redis or state explicitly that this is simulated.
- 8:00–9:00: Summarize findings and limitations.

Have a fallback recording/screenshots for a live-service outage, labelled as fallback evidence rather than proof that a service was live.

## 9. Acceptance gates (must be passed in order)

### Gate 0 — Repository and data contract
- Verify current branch, tracked files and latest commit.
- Keep original baseline unchanged.
- Confirm all processed tables and schemas/checksums needed by the chosen experiment.
- Identify which exact benchmark artifact produced every reported table.

### Gate 1 — Sequential benchmark integrity
- Re-run tests and benchmark from a clean checkout.
- Verify report sizes and all metric rows match generated metrics JSON.
- Check conversation ID separation and eligible prefix slicing.
- Validate confusion matrices and per-conversation prediction counts.
- Include length-only baseline and paired uncertainty tests.
- No strong sequence-order claim unless supported.

### Gate 2 — NoSQL integration and correctness
- Run mocks in tests and live-service integration tests separately.
- Verify idempotent seeding preserves moderator notes.
- Verify complete note CRUD and reference-data immutability.
- Verify saved review events are retrievable by conversation/horizon and chronological history is stable.
- Verify cache key, TTL, hit/miss, cache-failure behavior, and invalidation requirements.

### Gate 3 — Performance evidence
- Live MongoDB explain plan and actual index comparison.
- Live Redis cache test with real scoring function.
- Save raw benchmark outputs, configuration and P50/P95/P99 where sample size supports them.
- If services can't be provisioned, label as pending; do not substitute mock values.

### Gate 4 — Integrated live demo
- Clean launch from documented setup.
- Complete one journey from analysis -> saved review event -> moderator note -> retrieval/update -> repeat-score cache.
- Test startup failure and service unavailable states.
- Verify screenshot values match the version being demonstrated.

### Gate 5 — IEEE report lock
- At least 15 relevant research works verified and compared.
- All claims tied to actual outputs or cited literature.
- All visualizations generated from evidence.
- Clearly state dataset split history, internal holdout status, limitations, and which NoSQL results are simulated vs live.
- Confirm authorship metadata and remove secrets.
- Compile and visually inspect final IEEE PDF.

## 10. Explicitly prohibited claims

- Do not call the internal 80/20 split the official benchmark test set.
- Do not merge metrics from the 2,508-snapshot experiment and the earlier 4,188-partition experiment.
- Do not claim statistical significance when paired confidence intervals include zero.
- Do not say mock MongoDB/Redis is persistent.
- Do not describe synthetic delay as model inference.
- Do not claim an index improved performance without a live query-plan/benchmark comparison.
- Do not describe toxicity or a pattern support ratio as proof that a person committed an attack.
- Do not claim novelty based only on combining familiar algorithms.
- Do not state that the system predicts or prevents all attacks.
- Never commit tokens, credentials, or private deployment settings.

## 11. Worker protocol for Antigravity

Antigravity implements only the current approved task. It must:
1. Read this document first.
2. Inspect the current branch and existing code before editing.
3. Work only on `work/technical-direction` unless explicitly instructed otherwise.
4. Never modify `upgrade/nosql-presentation` or the user's immutable original local baseline as part of this task.
5. Never invent data, metrics, references, or successful service connections.
6. Preserve useful existing behavior and make small, reviewable commits.
7. For each task, return: files changed, commands run, exact test results, generated artifacts, known limitations, and a proposed diff/commit. Stop at each gate for independent review.
8. Never treat a passing mock test as live-service validation.
9. Do not merge or push to the baseline/default branch without explicit user approval.

## 12. Immediate first task

**Do not start model training or database optimization yet.** First perform a read-only repository audit on this branch:
- verify the current branch, HEAD commit and tree;
- verify latest benchmark code and docs match;
- map current code to acceptance gates;
- identify any mismatch among README, docs/RESULTS.md, test expectations, dashboard output, and generated artifacts;
- verify secret scanning status and note any historical leaked token without printing it;
- report findings, severity, evidence paths, and smallest next repair.
Make no source changes during this first audit.
