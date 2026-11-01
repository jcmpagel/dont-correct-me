from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scoring import policy_score, score_text

SOURCE_ORDER = [
    "SMHD",
    "Prolific DE",
    "Prolific EN",
    "PEER",
    "FH-SWF",
    "ASAP-SAS",
    "GNHK",
]
OCR_MODELS = {
    "Apple Vision OCR",
    "TrOCR Large Handwritten",
    "TrOCR Base Handwritten",
    "Tesseract LSTM",
    "PaddleOCR-VL 1.6",
}
COMPARISONS = [
    (
        "Gemini Pro--Claude Fable 5",
        "Gemini 3.1 Pro Preview",
        "Claude Fable 5",
        "exploratory",
    ),
    (
        "Gemini Pro--Gemini 3.5",
        "Gemini 3.1 Pro Preview",
        "Gemini 3.5 Flash",
        "exploratory",
    ),
    ("Gemini Pro--Gemma 31B", "Gemini 3.1 Pro Preview", "Gemma 4 31B", "exploratory"),
    ("Gemma 31B--Apple Vision", "Gemma 4 31B", "Apple Vision OCR", "exploratory"),
    ("Gemini 3.6--GPT-5.6 Sol", "Gemini 3.6 Flash", "GPT-5.6 Sol", "primary"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def accepted_alternatives(row: dict[str, str]) -> list[list[str]]:
    value = row.get("accepted_alternatives", "").strip()
    if not value:
        return []
    if len(row.get("canonical_corrections", "").split("|")) != 1:
        raise RuntimeError(
            "accepted_alternatives is only defined for single-target rows"
        )
    return [[item.strip() for item in value.split("|") if item.strip()]]


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return read_csv(path)


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: list[str] | None = None
) -> None:
    materialized = list(rows)
    fields = fields or (list(materialized[0]) if materialized else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def mean(values: Iterable[float | None]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    return sum(materialized) / len(materialized) if materialized else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolved = sum(int(row["resolved_target_count"]) for row in rows)
    unresolved = sum(int(row["unresolved_target_count"]) for row in rows)
    char_length = sum(int(row["char_reference_length"]) for row in rows)
    word_length = sum(int(row["word_reference_length"]) for row in rows)
    return {
        "n": len(rows),
        "successful_n": sum(int(row["successful"]) for row in rows),
        "failed_or_empty_n": sum(1 - int(row["successful"]) for row in rows),
        "cer_macro": mean(row["cer"] for row in rows),
        "wer_macro": mean(row["wer"] for row in rows),
        "cer_micro": sum(int(row["char_edit_distance"]) for row in rows) / char_length
        if char_length
        else None,
        "wer_micro": sum(int(row["word_edit_distance"]) for row in rows) / word_length
        if word_length
        else None,
        "exact": mean(row["exact_match"] for row in rows),
        "normalized_exact": mean(row["normalized_exact_match"] for row in rows),
        "preservation": sum(int(row["preserved_target_count"]) for row in rows)
        / resolved
        if resolved
        else None,
        "canonicalization": sum(int(row["canonicalized_target_count"]) for row in rows)
        / resolved
        if resolved
        else None,
        "other_substitution": sum(
            int(row["other_substitution_target_count"]) for row in rows
        )
        / resolved
        if resolved
        else None,
        "omission": sum(int(row["omitted_target_count"]) for row in rows) / resolved
        if resolved
        else None,
        "resolved_targets": resolved,
        "unresolved_targets": unresolved,
        "annotations": resolved + unresolved,
    }


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _successful(attempt: dict[str, Any]) -> int:
    value = attempt.get("successful", attempt.get("success", False))
    if isinstance(value, str):
        return int(value.lower() in {"1", "true", "yes"})
    return int(bool(value))


def reproduce(
    data_dir: Path,
    output_dir: Path,
    *,
    predictions_path: Path | None = None,
    seed: int = 27,
    resamples: int = 10_000,
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    sample_rows = read_csv(data_dir / "annotations.csv")
    if len(sample_rows) != 845:
        raise RuntimeError(f"Expected 845 samples, found {len(sample_rows)}")
    if len({row["sample_id"] for row in sample_rows}) != len(sample_rows):
        raise RuntimeError("Duplicate sample IDs in annotations.csv")
    samples = {
        row["sample_id"]: {
            "ground_truth": row["original_transcript"],
            "flagged_words": row["flagged_words"],
            "canonical_corrections": row["canonical_corrections"],
            "canonical_correction_variants": accepted_alternatives(row),
        }
        for row in sample_rows
    }
    sample_meta = {row["sample_id"]: row for row in sample_rows}
    frozen_workflow = predictions_path is None
    prediction_rows = read_predictions(predictions_path or data_dir / "predictions.csv")
    prediction_keys = [
        (str(row["model_name"]), str(row["sample_id"])) for row in prediction_rows
    ]
    if len(set(prediction_keys)) != len(prediction_keys):
        raise RuntimeError("Duplicate model/sample pairs in predictions")

    model_counts = Counter(name for name, _ in prediction_keys)
    if frozen_workflow and (
        len(model_counts) != 28 or set(model_counts.values()) != {845}
    ):
        raise RuntimeError(
            "Predictions must contain all 845 samples for each of the 28 models"
        )
    unknown_samples = {sample_id for _, sample_id in prediction_keys} - samples.keys()
    if unknown_samples:
        raise RuntimeError(f"Unknown samples in predictions: {sorted(unknown_samples)}")

    scored_rows: list[dict[str, Any]] = []
    for attempt in prediction_rows:
        sample_id = str(attempt["sample_id"])
        successful = _successful(attempt)
        hypothesis = str(attempt.get("prediction_text") or "") if successful else ""
        metrics = score_text(samples[sample_id], hypothesis)
        outcomes = Counter(item.outcome for item in metrics.target_outcomes)
        metadata = sample_meta[sample_id]
        row: dict[str, Any] = {
            "model_name": attempt["model_name"],
            "provider": attempt["provider"],
            "model": attempt["model"],
            "sample_id": sample_id,
            "source": metadata["source"],
            "language": metadata["language"],
            "cluster_id": metadata["cluster_id"],
            "successful": successful,
            "ground_truth": samples[sample_id]["ground_truth"],
            "prediction_text": hypothesis,
            "cer": metrics.cer,
            "wer": metrics.wer,
            "exact_match": int(metrics.exact_match),
            "normalized_exact_match": int(metrics.normalized_exact_match),
            "preserved_target_count": outcomes["preserved"],
            "canonicalized_target_count": outcomes["canonicalized"],
            "other_substitution_target_count": outcomes["other_substitution"],
            "omitted_target_count": outcomes["omitted"],
            "resolved_target_count": metrics.resolved_target_count,
            "unresolved_target_count": metrics.unresolved_target_count,
            "char_edit_distance": metrics.char_edit_distance,
            "char_reference_length": metrics.char_reference_length,
            "word_edit_distance": metrics.word_edit_distance,
            "word_reference_length": metrics.word_reference_length,
        }
        for label, punctuation, case in (
            ("punctuation", True, False),
            ("case", False, True),
            ("combined", True, True),
        ):
            cer, wer, exact = policy_score(
                samples[sample_id]["ground_truth"],
                hypothesis,
                ignore_punctuation=punctuation,
                ignore_case=case,
            )
            row[f"{label}_cer"], row[f"{label}_wer"], row[f"{label}_exact"] = (
                cer,
                wer,
                exact,
            )
        scored_rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_sample_scores.csv", scored_rows)

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_model[row["model_name"]].append(row)

    model_metrics = []
    for name, rows in by_model.items():
        metrics = {
            "model_name": name,
            "provider": rows[0]["provider"],
            "model": rows[0]["model"],
            "kind": "OCR" if name in OCR_MODELS else "VLM",
            **aggregate(rows),
        }
        for label in ("punctuation", "case", "combined"):
            metrics[f"{label}_cer_macro"] = mean(row[f"{label}_cer"] for row in rows)
            metrics[f"{label}_wer_macro"] = mean(row[f"{label}_wer"] for row in rows)
            metrics[f"{label}_exact"] = mean(row[f"{label}_exact"] for row in rows)
        model_metrics.append(metrics)
    model_metrics.sort(key=lambda row: (row["cer_macro"], row["model_name"]))
    write_csv(output_dir / "model_metrics.csv", model_metrics)

    breakdown_rows = []
    for dimension, values in (("source", SOURCE_ORDER), ("language", ["en", "de"])):
        for name, rows in by_model.items():
            for value in values:
                breakdown_rows.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "model_name": name,
                        **aggregate([row for row in rows if row[dimension] == value]),
                    }
                )
    write_csv(output_dir / "model_breakdowns.csv", breakdown_rows)

    policy_rows = []
    for row in model_metrics:
        for label in ("strict", "punctuation", "case", "combined"):
            policy_rows.append(
                {
                    "model_name": row["model_name"],
                    "policy": label,
                    "cer_macro": row["cer_macro"]
                    if label == "strict"
                    else row[f"{label}_cer_macro"],
                    "wer_macro": row["wer_macro"]
                    if label == "strict"
                    else row[f"{label}_wer_macro"],
                    "exact": row["exact"]
                    if label == "strict"
                    else row[f"{label}_exact"],
                }
            )
    write_csv(output_dir / "policy_sensitivity.csv", policy_rows)

    row_lookup = {(row["model_name"], row["sample_id"]): row for row in scored_rows}
    cluster_map: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    source_ids: dict[str, list[str]] = defaultdict(list)
    for sample_id, metadata in sample_meta.items():
        cluster_map[metadata["source"]][metadata["cluster_id"]].append(sample_id)
        source_ids[metadata["source"]].append(sample_id)
    complete_models = {
        name
        for name, rows in by_model.items()
        if {row["sample_id"] for row in rows} == set(samples)
    }
    rng = random.Random(seed)
    bootstrap_rows = []
    for label, model_a, model_b, status in COMPARISONS:
        if model_a not in complete_models or model_b not in complete_models:
            continue
        point = {
            "cer_macro": mean(
                row_lookup[(model_a, sample_id)]["cer"]
                - row_lookup[(model_b, sample_id)]["cer"]
                for sample_id in samples
            ),
            "exact": mean(
                row_lookup[(model_a, sample_id)]["exact_match"]
                - row_lookup[(model_b, sample_id)]["exact_match"]
                for sample_id in samples
            ),
        }
        total_resolved = sum(
            row_lookup[(model_a, sample_id)]["resolved_target_count"]
            for sample_id in samples
        )
        for metric, key in (
            ("preservation", "preserved_target_count"),
            ("canonicalization", "canonicalized_target_count"),
        ):
            point[metric] = (
                sum(row_lookup[(model_a, sample_id)][key] for sample_id in samples)
                - sum(row_lookup[(model_b, sample_id)][key] for sample_id in samples)
            ) / total_resolved
        distributions = {metric: [] for metric in point}
        for _ in range(resamples):
            replicate = {metric: 0.0 for metric in point}
            for source in SOURCE_ORDER:
                names = sorted(cluster_map[source])
                sampled_ids = [
                    sample_id
                    for cluster in rng.choices(names, k=len(names))
                    for sample_id in cluster_map[source][cluster]
                ]
                source_weight = len(source_ids[source]) / len(samples)
                for metric, key in (("cer_macro", "cer"), ("exact", "exact_match")):
                    replicate[metric] += source_weight * mean(
                        row_lookup[(model_a, sample_id)][key]
                        - row_lookup[(model_b, sample_id)][key]
                        for sample_id in sampled_ids
                    )
                target_weight = (
                    sum(
                        row_lookup[(model_a, sample_id)]["resolved_target_count"]
                        for sample_id in source_ids[source]
                    )
                    / total_resolved
                )
                for metric, key in (
                    ("preservation", "preserved_target_count"),
                    ("canonicalization", "canonicalized_target_count"),
                ):
                    rates = []
                    for model in (model_a, model_b):
                        denominator = sum(
                            row_lookup[(model, sample_id)]["resolved_target_count"]
                            for sample_id in sampled_ids
                        )
                        numerator = sum(
                            row_lookup[(model, sample_id)][key]
                            for sample_id in sampled_ids
                        )
                        rates.append(numerator / denominator)
                    replicate[metric] += target_weight * (rates[0] - rates[1])
            for metric in distributions:
                distributions[metric].append(replicate[metric])
        for metric, values in distributions.items():
            ordered = sorted(values)
            bootstrap_rows.append(
                {
                    "comparison": label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "claim_status": status,
                    "metric": metric,
                    "difference_a_minus_b": point[metric],
                    "ci_method": (
                        "95% percentile source-stratified paired cluster bootstrap"
                    ),
                    "ci_low": percentile(ordered, 0.025),
                    "ci_high": percentile(ordered, 0.975),
                    "seed": seed,
                    "resamples": resamples,
                }
            )
    bootstrap_fields = [
        "comparison",
        "model_a",
        "model_b",
        "claim_status",
        "metric",
        "difference_a_minus_b",
        "ci_method",
        "ci_low",
        "ci_high",
        "seed",
        "resamples",
    ]
    write_csv(output_dir / "bootstrap_results.csv", bootstrap_rows, bootstrap_fields)

    summary = {
        "seed": seed,
        "resamples": resamples,
        "counts": {
            "dataset_images": len(samples),
            "scored_samples": len({row["sample_id"] for row in scored_rows}),
            "annotations": sum(int(row["annotation_count"]) for row in sample_rows),
            "resolved_targets": sum(
                int(row["resolved_target_count"]) for row in sample_rows
            ),
            "unresolved_targets": sum(
                int(row["unresolved_target_count"]) for row in sample_rows
            ),
            "lanes": len(by_model),
        },
        "top_by_cer": model_metrics[:5],
        "top_by_preservation": sorted(
            model_metrics,
            key=lambda row: (
                row["preservation"] is not None,
                row["preservation"] or -1,
            ),
            reverse=True,
        )[:5],
        "primary_bootstrap": [
            row for row in bootstrap_rows if row["claim_status"] == "primary"
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
