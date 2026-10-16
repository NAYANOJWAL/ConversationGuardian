# Sequential benchmark results

This report is generated from the benchmark implementation and bundled processed data in this repository. It is intentionally explicit about the evaluation boundary so that internal holdout results are not mistaken for official CGA-Wiki test performance.

## Evaluation setup

| Item | Value |
|---|---|
| Task | Predict whether a discussion is attack-labelled from an initial chronological prefix |
| Data | Bundled `train_cleaned.parquet` + `train_sequences.parquet` snapshot |
| Conversations included | 2,508 |
| Training conversations | 2,006 |
| Internal holdout conversations | 502 |
| Attack-labelled prevalence | 50% train; 50% holdout |
| Split method | Stratified 80/20 split by `conversation_id` |
| Random seed | 42 |
| Train/holdout conversation-ID overlap | **0** |
| Decision threshold | Fixed at 0.5; not tuned on the holdout |
| Workflow run commit | `abaaddba1460c699830b0200a5fec1e876eb228c` (benchmark script unchanged since `7242cfafa51c58d52e87575e1904d2a523dbde1c`) |
| Successful GitHub Actions run | [Run 38862671743](https://github.com/NAYANOJWAL/ConversationGuardian/actions/runs/38862671743) |
| Downloadable JSON/CSV/prediction artifact | [Benchmark artifact](https://github.com/NAYANOJWAL/ConversationGuardian/actions/runs/38862671743/artifacts/12029691970) |

The source snapshot contains only rows marked `train`; the 502 conversations are an **internal holdout sampled from that bundled training snapshot**, not an untouched official dataset test partition. Conversations, rather than comments, are the split unit so that turns from one conversation cannot appear in both training and holdout.

## Results by prefix horizon

The 25%, 50%, and 75% rows represent early-warning horizons. The 100% row is a full-context reference and must not be described as an early-warning score.

| Model | Prefix | Accuracy | Macro-F1 | Attack recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority-class reference | Reference | 0.5000 | 0.3333 | 0.0000 | 0.5000 | 0.5000 |
| TF-IDF text baseline | 25% | 0.5438 | 0.5437 | **0.5618** | 0.5400 | 0.5670 |
| Word + character TF-IDF | 25% | 0.5418 | 0.5418 | 0.5299 | **0.5538** | 0.5730 |
| State-frequency (orderless) | 25% | 0.5219 | 0.5145 | 0.3984 | 0.5396 | 0.5382 |
| Ordered-state n-gram | 25% | 0.5458 | 0.5193 | 0.3108 | 0.5266 | 0.5272 |
| Hybrid text + state sequence | 25% | **0.5657** | **0.5647** | 0.5179 | 0.5531 | **0.5808** |
| TF-IDF text baseline | 50% | 0.5677 | 0.5671 | **0.6056** | **0.5718** | **0.5925** |
| Word + character TF-IDF | 50% | 0.5538 | 0.5534 | 0.5817 | 0.5700 | 0.5903 |
| State-frequency (orderless) | 50% | 0.5478 | 0.5478 | 0.5538 | 0.5398 | 0.5615 |
| Ordered-state n-gram | 50% | 0.4920 | 0.4919 | 0.5100 | 0.5302 | 0.5329 |
| Hybrid text + state sequence | 50% | **0.5737** | **0.5735** | 0.5976 | 0.5682 | 0.5895 |
| TF-IDF text baseline | 75% | **0.6255** | **0.6254** | **0.6375** | 0.6454 | **0.6644** |
| Word + character TF-IDF | 75% | 0.6116 | 0.6115 | 0.6215 | 0.6451 | 0.6564 |
| State-frequency (orderless) | 75% | 0.5518 | 0.5507 | 0.6016 | 0.5995 | 0.5842 |
| Ordered-state n-gram | 75% | 0.5279 | 0.5278 | 0.5179 | 0.5935 | 0.5670 |
| Hybrid text + state sequence | 75% | 0.6116 | 0.6115 | 0.6016 | **0.6556** | 0.6445 |
| TF-IDF text baseline | 100% | 0.7092 | 0.7091 | **0.7211** | 0.7941 | 0.7993 |
| Word + character TF-IDF | 100% | 0.7231 | 0.7231 | 0.7171 | 0.8141 | 0.8166 |
| State-frequency (orderless) | 100% | 0.7709 | 0.7596 | 0.5538 | 0.8692 | 0.8232 |
| Ordered-state n-gram | 100% | 0.7590 | 0.7508 | 0.5777 | 0.8669 | 0.8208 |
| Hybrid text + state sequence | 100% | **0.8127** | **0.8098** | 0.6892 | **0.9081** | **0.8834** |

*Metrics reproduced from successful CI run 38862671743. Scores are rounded to four decimal places. The downloadable artifact contains the full JSON metrics, confusion matrices, bootstrap intervals, and holdout prediction probabilities.*

## What the results support

1. **At 25%, the signal is limited.** The hybrid model has Macro-F1 0.5647 and PR-AUC 0.5531. This is better than the majority reference, but it is not strong enough to justify automatic enforcement or confident claims that attacks can reliably be predicted from the opening turns.
2. **More of the conversation helps.** At 75%, the word TF-IDF baseline has the highest Macro-F1 (0.6254), while the hybrid model has the highest PR-AUC (0.6556). The best model therefore depends on the metric and the warning horizon.
3. **Full-context performance is not an early-warning result.** The hybrid's Macro-F1 reaches 0.8098 and PR-AUC 0.9081 at 100%, but this includes the entire conversation and may include the content that explains the label.
4. **Sequence-only features are not automatically better.** The ordered-state n-gram classifier does not outperform text or hybrid models consistently. The state-frequency ablation is included to test whether ordering adds value rather than assuming it does.

## Limitations and next experiments

- This is a single deterministic, stratified internal holdout from the bundled processed training snapshot. External validation and repeated splits are still needed.
- The models use a fixed 0.5 threshold. The reported scores are not threshold-calibrated operating points.
- Bootstrap intervals are included in the JSON artifact; they describe sampling uncertainty for this holdout and do not correct for dataset shift or model-selection bias.
- The benchmark classifiers are separate from the DistilBERT full-thread score in the Risk Simulator. Do not mix their metrics or imply that this benchmark validates the DistilBERT checkpoint.
- A sensible next experiment is to validate on a genuinely untouched conversation set and evaluate threshold choice on a separate validation split before discussing deployment.

**Recommended assessment claim:** “We evaluated conversation-level early warning at multiple prefix lengths against majority, text, and state-sequence baselines. The results show limited predictive signal in very early turns and stronger classification with more context, so the tool is designed for moderator triage rather than automatic decisions.”
