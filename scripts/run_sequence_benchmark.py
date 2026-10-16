"""Reproducible conversation-level early-warning benchmark.

This experiment uses the bundled processed training snapshot only. It creates a
stratified holdout by conversation_id (never by comment), evaluates lexical and
symbol-sequence models on matching prefix fractions, and writes auditable
metrics/predictions. It is NOT the official CGA-Wiki held-out test benchmark.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 42
PREFIX_FRACTIONS = (0.25, 0.50, 0.75, 1.0)


def _to_binary_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer, float, np.floating)) and not pd.isna(value):
        if float(value) in (0.0, 1.0):
            return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "positive", "attack", "derailed"}:
            return 1
        if normalized in {"0", "false", "no", "negative", "healthy", "not_attack"}:
            return 0
    raise ValueError(f"Unexpected has_attack label: {value!r}")


def normalize_sequence(value: Any) -> list[str]:
    """Normalize a Parquet list/array or a serialized list to ordered state tokens."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                parsed = raw.split()
        value = parsed
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [str(value)]
    return [str(token).strip() for token in value if str(token).strip()]


def take_prefix(items: list[str], fraction: float) -> list[str]:
    """Return the first ceil(fraction * length) events, with no future events."""
    if not 0 < fraction <= 1:
        raise ValueError("Prefix fraction must be in (0, 1].")
    if not items:
        return []
    length = min(len(items), max(1, math.ceil(len(items) * fraction)))
    return items[:length]


def load_conversation_table(cleaned_path: Path, sequences_path: Path) -> pd.DataFrame:
    cleaned = pd.read_parquet(cleaned_path)
    sequences = pd.read_parquet(sequences_path)

    required_clean = {"conversation_id", "has_attack"}
    required_seq = {"conversation_id", "symbol_sequence"}
    if missing := required_clean - set(cleaned.columns):
        raise ValueError(f"Cleaned data missing required columns: {sorted(missing)}")
    if missing := required_seq - set(sequences.columns):
        raise ValueError(f"Sequence data missing required columns: {sorted(missing)}")

    text_column = "clean_text" if "clean_text" in cleaned.columns else "text"
    if text_column not in cleaned.columns:
        raise ValueError("Cleaned data must contain either clean_text or text.")
    order_column = "utterance_index" if "utterance_index" in cleaned.columns else None

    cleaned = cleaned.copy()
    cleaned["conversation_id"] = cleaned["conversation_id"].astype(str)
    sequences = sequences.copy()
    sequences["conversation_id"] = sequences["conversation_id"].astype(str)
    if sequences["conversation_id"].duplicated().any():
        raise ValueError("Sequence table must contain at most one row per conversation_id.")

    sequence_map = {
        str(row.conversation_id): normalize_sequence(row.symbol_sequence)
        for row in sequences[["conversation_id", "symbol_sequence"]].itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    for conversation_id, group in cleaned.groupby("conversation_id", sort=True):
        labels = {_to_binary_label(x) for x in group["has_attack"].dropna().unique()}
        if len(labels) != 1:
            raise ValueError(
                f"Conversation {conversation_id!r} has inconsistent has_attack labels."
            )
        if order_column:
            group = group.sort_values(order_column, kind="stable")
        messages = [
            str(text).strip()
            for text in group[text_column].tolist()
            if isinstance(text, str) and text.strip()
        ]
        states = sequence_map.get(conversation_id, [])
        if not messages or not states:
            continue
        records.append(
            {
                "conversation_id": conversation_id,
                "label": next(iter(labels)),
                "messages": messages,
                "states": states,
                "message_count": len(messages),
                "state_count": len(states),
                "source_split": str(group["split"].iloc[0])
                if "split" in group.columns
                else "unspecified",
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No conversations had both messages and symbol sequences.")
    if frame["label"].nunique() != 2:
        raise ValueError("The processed snapshot must contain both attack and healthy conversations.")
    return frame


def evaluate_predictions(y_true: np.ndarray, probability: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    predicted = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1]).tolist()
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", labels=[0, 1], zero_division=0)),
        "attack_precision": float(precision_score(y_true, predicted, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(y_true, predicted, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probability))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return metrics, {"confusion_matrix_labels": ["healthy (0)", "attack (1)"], "confusion_matrix": matrix}


def _make_text_model():
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=SEED,
            solver="liblinear",
        ),
    )


def _make_state_frequency_model():
    """Orderless ablation: use the same state vocabulary but ignore transition order."""
    return make_pipeline(
        TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 1),
            min_df=1,
            max_features=20000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=SEED,
            solver="liblinear",
        ),
    )


def _make_sequence_model():
    return make_pipeline(
        TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 3),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=SEED,
            solver="liblinear",
        ),
    )


def _positive_probability(model, values: list[str]) -> np.ndarray:
    probabilities = model.predict_proba(values)
    classes = list(model[-1].classes_)
    positive_column = classes.index(1)
    return probabilities[:, positive_column]


def _bootstrap_macro_f1(
    y_true: np.ndarray, probability: np.ndarray, samples: int = 500
) -> list[float]:
    """Conversation-level percentile interval; fixed seed makes the report reproducible."""
    rng = np.random.default_rng(SEED)
    predicted = (probability >= 0.5).astype(int)
    values = []
    n = len(y_true)
    for _ in range(samples):
        indices = rng.integers(0, n, size=n)
        values.append(
            float(
                f1_score(
                    y_true[indices],
                    predicted[indices],
                    average="macro",
                    labels=[0, 1],
                    zero_division=0,
                )
            )
        )
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def run_benchmark(frame: pd.DataFrame, test_size: float = 0.20) -> tuple[dict[str, Any], pd.DataFrame]:
    y = frame["label"].to_numpy(dtype=int)
    indexes = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indexes,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    y_train = train["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)

    results: list[dict[str, Any]] = []
    predictions = pd.DataFrame(
        {"conversation_id": test["conversation_id"], "label": y_test}
    )

    # Majority-class reference keeps the baseline honest and easy to interpret.
    majority_label = int(pd.Series(y_train).value_counts().idxmax())
    majority_probability = np.full(len(test), 0.99 if majority_label else 0.01)
    metrics, extra = evaluate_predictions(y_test, majority_probability)
    results.append(
        {
            "model": "Majority-class reference",
            "prefix_fraction": None,
            "prefix_label": "reference",
            "n_train": len(train),
            "n_test": len(test),
            **metrics,
            "macro_f1_bootstrap_95ci": _bootstrap_macro_f1(y_test, majority_probability),
            **extra,
        }
    )
    predictions["majority_reference_probability"] = majority_probability

    for fraction in PREFIX_FRACTIONS:
        prefix_label = f"{fraction:.0%}"
        train_text = [
            " ".join(take_prefix(row.messages, fraction))
            for row in train.itertuples(index=False)
        ]
        test_text = [
            " ".join(take_prefix(row.messages, fraction))
            for row in test.itertuples(index=False)
        ]
        text_model = _make_text_model()
        text_model.fit(train_text, y_train)
        text_probability = _positive_probability(text_model, test_text)
        metrics, extra = evaluate_predictions(y_test, text_probability)
        results.append(
            {
                "model": "TF-IDF text baseline",
                "prefix_fraction": fraction,
                "prefix_label": prefix_label,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
                "macro_f1_bootstrap_95ci": _bootstrap_macro_f1(y_test, text_probability),
                **extra,
            }
        )
        predictions[f"text_{int(fraction * 100)}pct_probability"] = text_probability

        train_states = [
            " ".join(take_prefix(row.states, fraction))
            for row in train.itertuples(index=False)
        ]
        test_states = [
            " ".join(take_prefix(row.states, fraction))
            for row in test.itertuples(index=False)
        ]

        # Ablation: same states, but no transition/order information.
        frequency_model = _make_state_frequency_model()
        frequency_model.fit(train_states, y_train)
        frequency_probability = _positive_probability(frequency_model, test_states)
        metrics, extra = evaluate_predictions(y_test, frequency_probability)
        results.append(
            {
                "model": "State-frequency ablation (orderless)",
                "prefix_fraction": fraction,
                "prefix_label": prefix_label,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
                "macro_f1_bootstrap_95ci": _bootstrap_macro_f1(y_test, frequency_probability),
                **extra,
            }
        )
        predictions[f"statefreq_{int(fraction * 100)}pct_probability"] = frequency_probability

        sequence_model = _make_sequence_model()
        sequence_model.fit(train_states, y_train)
        sequence_probability = _positive_probability(sequence_model, test_states)
        metrics, extra = evaluate_predictions(y_test, sequence_probability)
        results.append(
            {
                "model": "Ordered-state n-gram model",
                "prefix_fraction": fraction,
                "prefix_label": prefix_label,
                "n_train": len(train),
                "n_test": len(test),
                **metrics,
                "macro_f1_bootstrap_95ci": _bootstrap_macro_f1(y_test, sequence_probability),
                **extra,
            }
        )
        predictions[f"sequence_{int(fraction * 100)}pct_probability"] = sequence_probability

    metric_frame = pd.DataFrame(results)
    best_early = metric_frame[
        (metric_frame["model"] == "Ordered-state n-gram model")
        & (metric_frame["prefix_fraction"].isin([0.25, 0.50, 0.75]))
    ].sort_values("macro_f1", ascending=False)
    best_model = (
        {
            "model": str(best_early.iloc[0]["model"]),
            "prefix_fraction": float(best_early.iloc[0]["prefix_fraction"]),
            "macro_f1": float(best_early.iloc[0]["macro_f1"]),
        }
        if not best_early.empty
        else None
    )

    report: dict[str, Any] = {
        "title": "ConversationGuardian early-warning sequence benchmark",
        "purpose": "Predict whether a conversation has an attack using only an initial chronological prefix.",
        "data_source": "Bundled processed train_cleaned.parquet + train_sequences.parquet",
        "evaluation_protocol": (
            "Deterministic stratified 80/20 holdout split at conversation_id, seed=42. "
            "All comments in a conversation remain in one split. Each model is fitted only on training conversations. "
            "Prediction threshold is fixed at 0.5; no threshold is tuned on the holdout."
        ),
        "interpretation_limit": (
            "This is an internal holdout from the repository's bundled processed training snapshot, "
            "not the official benchmark test set and not a replacement for external validation."
        ),
        "random_seed": SEED,
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "n_conversations": int(len(frame)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_attack_prevalence": float(y_train.mean()),
        "test_attack_prevalence": float(y_test.mean()),
        "positive_label": "has_attack=1",
        # Keep None (for the reference model's prefix fraction) as JSON null.
        # Passing through a DataFrame would coerce that mixed column to NaN.
        "metrics": results,
        "best_early_sequence_model_by_macro_f1": best_model,
    }
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=Path("data/cleaned/train_cleaned.parquet"))
    parser.add_argument("--sequences", type=Path, default=Path("data/sequences/train_sequences.parquet"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    args = parser.parse_args()

    frame = load_conversation_table(args.cleaned, args.sequences)
    report, predictions = run_benchmark(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "sequence_benchmark_metrics.json"
    csv_path = args.output / "sequence_benchmark_predictions.csv"
    metrics_path = args.output / "sequence_benchmark_summary.csv"

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    predictions.to_csv(csv_path, index=False)
    pd.DataFrame(report["metrics"]).drop(
        columns=["confusion_matrix", "macro_f1_bootstrap_95ci"], errors="ignore"
    ).to_csv(metrics_path, index=False)

    summary_columns = [
        "model",
        "prefix_label",
        "accuracy",
        "macro_f1",
        "attack_f1",
        "attack_recall",
        "pr_auc",
        "roc_auc",
    ]
    summary = pd.DataFrame(report["metrics"])
    print(json.dumps({k: report[k] for k in (
        "title", "data_source", "evaluation_protocol", "interpretation_limit",
        "n_conversations", "n_train", "n_test", "train_attack_prevalence",
        "test_attack_prevalence", "best_early_sequence_model_by_macro_f1",
    )}, indent=2))
    print("\nRESULTS (threshold=0.5; conversation-level holdout)")
    print(summary[summary_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWrote: {json_path}\nWrote: {metrics_path}\nWrote: {csv_path}")


if __name__ == "__main__":
    main()
