import csv
import difflib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import _lexicality as lexicality
from _benchmark import OCR_MODELS, read_csv

ROOT = Path(__file__).resolve().parents[1]


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        for position in range(start, end):
            result[ordered[position][0]] = (start + 1 + end) / 2
        start = end
    return result


def correlation(rows: list[dict], x: str, y: str) -> float:
    return statistics.correlation(
        ranks([row[x] for row in rows]), ranks([row[y] for row in rows])
    )


def read_indexed_csv(
    path: Path, key: str, expected_count: int
) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    indexed = {}
    for row in rows:
        identifier = row[key]
        if not identifier or identifier in indexed:
            raise ValueError(f"Empty or duplicate {key} in {path.name}: {identifier!r}")
        indexed[identifier] = row
    if len(indexed) != expected_count:
        raise ValueError(
            f"Expected {expected_count} rows in {path.name}, found {len(indexed)}"
        )
    return indexed


def main(root: Path = ROOT) -> None:
    benchmark_path = root / "results/benchmark/model_metrics.csv"
    if not benchmark_path.exists():
        raise SystemExit("Run python evaluation/benchmark.py first.")
    benchmark = read_indexed_csv(benchmark_path, "model_name", 28)
    samples = read_indexed_csv(root / "nonwords/annotations.csv", "sample_id", 1000)
    words = lexicality.load_scowl(root / "evaluation/lexicon/scowl.txt")
    if len(words) != 90392:
        raise ValueError(f"Expected 90,392 SCOWL words, found {len(words)}")
    reference = lexicality.aggregate(
        [
            lexicality.classify(row["original_transcript"], words.__contains__)
            for row in samples.values()
        ]
    )
    if reference["tokens"] != 5500 or reference["known_tokens"] != 0:
        raise ValueError(
            "Nonword references must contain 5,500 tokens and no SCOWL words"
        )
    reference_tokens = {
        sample_id: lexicality.tokenize(row["original_transcript"])
        for sample_id, row in samples.items()
    }
    groups = defaultdict(list)
    preserved = defaultdict(int)
    errors = defaultdict(int)
    seen = set()
    for row in read_csv(root / "nonwords/predictions.csv"):
        name, sample_id = row["model_name"], row["sample_id"]
        if name not in benchmark:
            raise ValueError(f"Model missing from benchmark results: {name}")
        if sample_id not in samples:
            raise ValueError(f"Unknown sample in predictions: {sample_id}")
        if (name, sample_id) in seen:
            raise ValueError(f"Duplicate prediction: {name}, {sample_id}")
        seen.add((name, sample_id))
        text = row["prediction_text"]
        groups[name].append(lexicality.classify(text, words.__contains__))
        errors[name] += bool(row["error"]) or not text.strip()
        matches = difflib.SequenceMatcher(
            a=reference_tokens[sample_id],
            b=lexicality.tokenize(text),
            autojunk=False,
        ).get_matching_blocks()
        preserved[name] += sum(length for _, _, length in matches)
    if set(groups) != set(benchmark) or any(
        len(rows) != len(samples) for rows in groups.values()
    ):
        raise ValueError(
            "Predictions must contain all 1,000 samples for each of the 28 models"
        )
    results = []
    for name, rows in groups.items():
        counts = lexicality.aggregate(rows)
        if counts["tokens"] == 0:
            raise ValueError(
                f"Real-word rate is undefined: no output tokens for {name}"
            )
        results.append(
            {
                "model": name,
                "errors": errors[name],
                "kind": "OCR" if name in OCR_MODELS else "VLM",
                "reference_tokens": reference["tokens"],
                "preserved": preserved[name],
                "nonword_preservation_pct": 100 * preserved[name] / reference["tokens"],
                "real_word_rate_pct": 100 * counts["known_tokens"] / counts["tokens"],
                "real_word_tokens": counts["known_tokens"],
                "output_tokens": counts["tokens"],
                "all_real_lines": counts["all_known_lines"],
                "any_real_lines": counts["lines_with_at_least_one_known"],
                "spelling_preservation_pct": float(benchmark[name]["preservation"])
                * 100,
                "spelling_overcorrection_pct": float(
                    benchmark[name]["canonicalization"]
                )
                * 100,
            }
        )
    correlations = {
        "nonword_vs_spelling_preservation": correlation(
            results, "nonword_preservation_pct", "spelling_preservation_pct"
        ),
        "lexicality_vs_overcorrection_all": correlation(
            results, "real_word_rate_pct", "spelling_overcorrection_pct"
        ),
        "lexicality_vs_overcorrection_vlm": correlation(
            [row for row in results if row["kind"] == "VLM"],
            "real_word_rate_pct",
            "spelling_overcorrection_pct",
        ),
    }
    out = root / "results/nonwords"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["model"]))
    (out / "correlations.json").write_text(
        json.dumps(correlations, indent=2) + "\n", encoding="utf-8"
    )
    print("Evaluated 28 models on 1,000 nonword lines.")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
