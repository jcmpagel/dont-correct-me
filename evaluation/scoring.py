from __future__ import annotations

import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetSpanOutcome:
    target: str
    canonical_forms: tuple[str, ...]
    visible_reference: str | None
    hypothesis_span: str | None
    outcome: str
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class TextMetrics:
    cer: float
    wer: float
    exact_match: bool
    normalized_exact_match: bool
    spelling_preservation_rate: float | None
    spelling_normalization_rate: float | None
    spelling_other_substitution_rate: float | None
    spelling_omission_rate: float | None
    resolved_target_count: int
    unresolved_target_count: int
    char_edit_distance: int
    char_reference_length: int
    word_edit_distance: int
    word_reference_length: int
    target_outcomes: tuple[TargetSpanOutcome, ...]


def normalize_text(text: str) -> str:
    normalized = (
        unicodedata.normalize("NFC", text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", " ")
    )
    return re.sub(r"[^\S\n]+(?=[,.;:!?–—…-])", "", normalized)


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for index, ref_item in enumerate(reference, 1):
        current = [index]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def score_text(sample: dict[str, Any], hypothesis: str) -> TextMetrics:
    reference = normalize_text(str(sample.get("ground_truth", "")))
    hypothesis = normalize_text(hypothesis)
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    char_edit_distance = levenshtein(ref_chars, hyp_chars)
    word_edit_distance = levenshtein(ref_words, hyp_words)
    flagged = _split_compact(sample.get("flagged_words", ""))
    corrections = _split_compact(sample.get("canonical_corrections", ""))
    alignment = (
        _minimum_edit_alignment(reference, hypothesis, char_edit_distance)
        if flagged
        else []
    )
    target_outcomes: list[TargetSpanOutcome] = []
    for index, word in enumerate(flagged):
        correction = corrections[index] if index < len(corrections) else ""
        canonical_forms = tuple(_canonical_forms(sample, index, correction))
        occurrences = _target_occurrences(reference, word)
        if len(occurrences) != 1:
            target_outcomes.append(
                TargetSpanOutcome(
                    target=word,
                    canonical_forms=canonical_forms,
                    visible_reference=None,
                    hypothesis_span=None,
                    outcome="unresolved",
                    unresolved_reason=(
                        "target_absent_from_reference"
                        if not occurrences
                        else "target_repeated_in_reference"
                    ),
                )
            )
            continue
        start, end = occurrences[0]
        visible_reference = reference[start:end]
        hypothesis_span = _project_reference_span(
            alignment, hypothesis, start, end, reference
        )
        comparable_hypothesis = unicodedata.normalize("NFC", hypothesis_span).casefold()
        if (
            comparable_hypothesis
            == unicodedata.normalize("NFC", visible_reference).casefold()
        ):
            outcome = "preserved"
        elif any(
            comparable_hypothesis == unicodedata.normalize("NFC", form).casefold()
            for form in canonical_forms
            if form
        ):
            outcome = "canonicalized"
        elif hypothesis_span == "":
            outcome = "omitted"
        else:
            outcome = "other_substitution"
        target_outcomes.append(
            TargetSpanOutcome(
                target=word,
                canonical_forms=canonical_forms,
                visible_reference=visible_reference,
                hypothesis_span=hypothesis_span,
                outcome=outcome,
            )
        )
    resolved = [item for item in target_outcomes if item.outcome != "unresolved"]

    def rate(name: str) -> float | None:
        return (
            sum(item.outcome == name for item in resolved) / len(resolved)
            if resolved
            else None
        )

    return TextMetrics(
        cer=char_edit_distance / len(ref_chars)
        if ref_chars
        else float(bool(hyp_chars)),
        wer=word_edit_distance / len(ref_words)
        if ref_words
        else float(bool(hyp_words)),
        exact_match=ref_chars == hyp_chars,
        normalized_exact_match=[char.casefold() for char in ref_chars]
        == [char.casefold() for char in hyp_chars],
        spelling_preservation_rate=rate("preserved"),
        spelling_normalization_rate=rate("canonicalized"),
        spelling_other_substitution_rate=rate("other_substitution"),
        spelling_omission_rate=rate("omitted"),
        resolved_target_count=len(resolved),
        unresolved_target_count=len(target_outcomes) - len(resolved),
        char_edit_distance=char_edit_distance,
        char_reference_length=len(ref_chars),
        word_edit_distance=word_edit_distance,
        word_reference_length=len(ref_words),
        target_outcomes=tuple(target_outcomes),
    )


def apply_policy(text: str, *, ignore_punctuation: bool, ignore_case: bool) -> str:
    value = normalize_text(text)
    if ignore_punctuation:
        value = "".join(
            character
            for character in value
            if not unicodedata.category(character).startswith("P")
        )
        value = re.sub(r"\s+", " ", value).strip()
    if ignore_case:
        value = value.casefold()
    return value


def policy_score(
    reference: str, hypothesis: str, *, ignore_punctuation: bool, ignore_case: bool
) -> tuple[float, float, float]:
    ref = apply_policy(
        reference,
        ignore_punctuation=ignore_punctuation,
        ignore_case=ignore_case,
    )
    hyp = apply_policy(
        hypothesis,
        ignore_punctuation=ignore_punctuation,
        ignore_case=ignore_case,
    )
    ref_words, hyp_words = ref.split(), hyp.split()
    cer = levenshtein(list(ref), list(hyp)) / len(ref) if ref else float(bool(hyp))
    wer = (
        levenshtein(ref_words, hyp_words) / len(ref_words)
        if ref_words
        else float(bool(hyp_words))
    )
    return cer, wer, float(ref == hyp)


def _split_compact(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def _canonical_forms(sample: dict[str, Any], index: int, primary: str) -> list[str]:
    variants = sample.get("canonical_correction_variants", [])
    if isinstance(variants, str):
        variants = json.loads(variants or "[]")
    forms = [primary.strip()] if primary.strip() else []
    if (
        isinstance(variants, list)
        and index < len(variants)
        and isinstance(variants[index], list)
    ):
        forms.extend(
            str(value).strip() for value in variants[index] if str(value).strip()
        )
    return list(dict.fromkeys(forms))


def _target_occurrences(text: str, target: str) -> list[tuple[int, int]]:
    return [
        match.span()
        for match in re.finditer(
            rf"(?<!\w){re.escape(target)}(?!\w)", text, flags=re.IGNORECASE
        )
    ]


def _project_reference_span(
    alignment: list[tuple[int | None, int | None]],
    hypothesis: str,
    start: int,
    end: int,
    reference: str,
) -> str:
    """Project a target and include lexical additions at its word boundaries.

    Character alignment alone excludes insertions exactly at the target edges.
    Expanding the projected interval to complete words prevents a changed word
    (e.g. ``fü`` -> ``für``) from being counted as literally preserved. Existing
    reference affixes outside the annotation remain outside the target.
    """
    projected: list[int] = []
    reference_cursor = 0
    for reference_index, hypothesis_index in alignment:
        if reference_index is None:
            if start < reference_cursor < end and hypothesis_index is not None:
                projected.append(hypothesis_index)
            continue
        if start <= reference_index < end and hypothesis_index is not None:
            projected.append(hypothesis_index)
        reference_cursor = reference_index + 1
    if not projected:
        return ""
    left, right = min(projected), max(projected) + 1
    while (
        left
        and _lexical_character(hypothesis[left])
        and _lexical_character(hypothesis[left - 1])
    ):
        left -= 1
    while (
        right < len(hypothesis)
        and _lexical_character(hypothesis[right - 1])
        and _lexical_character(hypothesis[right])
    ):
        right += 1
    connectors = "'-’"
    left_join = not (
        start >= 2
        and reference[start - 1] in connectors
        and _lexical_character(reference[start - 2])
    )
    right_join = not (
        end + 1 < len(reference)
        and reference[end] in connectors
        and _lexical_character(reference[end + 1])
    )
    while (
        left_join
        and left >= 2
        and _lexical_character(hypothesis[left])
        and hypothesis[left - 1] in connectors
        and _lexical_character(hypothesis[left - 2])
    ):
        left -= 2
        while left and _lexical_character(hypothesis[left - 1]):
            left -= 1
    while (
        right_join
        and right + 1 < len(hypothesis)
        and _lexical_character(hypothesis[right - 1])
        and hypothesis[right] in connectors
        and _lexical_character(hypothesis[right + 1])
    ):
        right += 2
        while right < len(hypothesis) and _lexical_character(hypothesis[right]):
            right += 1
    return hypothesis[left:right]


def _lexical_character(character: str) -> bool:
    return (
        character.isalnum()
        or character == "_"
        or unicodedata.category(character).startswith("M")
    )


def _minimum_edit_alignment(
    reference: str, hypothesis: str, expected_distance: int | None = None
) -> list[tuple[int | None, int | None]]:
    candidate = _sequence_matcher_alignment(reference, hypothesis)
    distance = (
        expected_distance
        if expected_distance is not None
        else levenshtein(list(reference), list(hypothesis))
    )
    if _alignment_cost(reference, hypothesis, candidate) == distance:
        return candidate
    return _hirschberg_alignment(reference, hypothesis)


def _sequence_matcher_alignment(
    reference: str, hypothesis: str
) -> list[tuple[int | None, int | None]]:
    alignment: list[tuple[int | None, int | None]] = []
    matcher = difflib.SequenceMatcher(a=reference, b=hypothesis, autojunk=False)
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag == "equal":
            alignment.extend(
                (ref_start + offset, hyp_start + offset)
                for offset in range(ref_end - ref_start)
            )
        elif tag == "delete":
            alignment.extend((index, None) for index in range(ref_start, ref_end))
        elif tag == "insert":
            alignment.extend((None, index) for index in range(hyp_start, hyp_end))
        else:
            shared = min(ref_end - ref_start, hyp_end - hyp_start)
            alignment.extend(
                (ref_start + offset, hyp_start + offset) for offset in range(shared)
            )
            alignment.extend(
                (index, None) for index in range(ref_start + shared, ref_end)
            )
            alignment.extend(
                (None, index) for index in range(hyp_start + shared, hyp_end)
            )
    return alignment


def _alignment_cost(
    reference: str,
    hypothesis: str,
    alignment: list[tuple[int | None, int | None]],
) -> int:
    return sum(
        1
        for reference_index, hypothesis_index in alignment
        if reference_index is None
        or hypothesis_index is None
        or reference[reference_index] != hypothesis[hypothesis_index]
    )


def _last_distance_row(reference: str, hypothesis: str) -> list[int]:
    previous = list(range(len(hypothesis) + 1))
    for reference_character in reference:
        current = [previous[0] + 1]
        for index, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous


def _small_minimum_alignment(
    reference: str,
    hypothesis: str,
    reference_offset: int = 0,
    hypothesis_offset: int = 0,
) -> list[tuple[int | None, int | None]]:
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distances[row][0] = row
    for column in range(columns):
        distances[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            distances[row][column] = min(
                distances[row - 1][column - 1]
                + (reference[row - 1] != hypothesis[column - 1]),
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
            )
    alignment: list[tuple[int | None, int | None]] = []
    row, column = len(reference), len(hypothesis)
    while row or column:
        if (
            row
            and column
            and distances[row][column]
            == distances[row - 1][column - 1]
            + (reference[row - 1] != hypothesis[column - 1])
        ):
            alignment.append(
                (reference_offset + row - 1, hypothesis_offset + column - 1)
            )
            row -= 1
            column -= 1
        elif row and distances[row][column] == distances[row - 1][column] + 1:
            alignment.append((reference_offset + row - 1, None))
            row -= 1
        else:
            alignment.append((None, hypothesis_offset + column - 1))
            column -= 1
    alignment.reverse()
    return alignment


def _hirschberg_alignment(
    reference: str,
    hypothesis: str,
    reference_offset: int = 0,
    hypothesis_offset: int = 0,
) -> list[tuple[int | None, int | None]]:
    if not reference:
        return [(None, hypothesis_offset + index) for index in range(len(hypothesis))]
    if not hypothesis:
        return [(reference_offset + index, None) for index in range(len(reference))]
    if len(reference) <= 2 or len(hypothesis) <= 2:
        return _small_minimum_alignment(
            reference, hypothesis, reference_offset, hypothesis_offset
        )
    midpoint = len(reference) // 2
    left_scores = _last_distance_row(reference[:midpoint], hypothesis)
    right_scores = _last_distance_row(reference[midpoint:][::-1], hypothesis[::-1])
    split = min(
        range(len(hypothesis) + 1),
        key=lambda index: (
            left_scores[index] + right_scores[len(hypothesis) - index],
            index,
        ),
    )
    return _hirschberg_alignment(
        reference[:midpoint],
        hypothesis[:split],
        reference_offset,
        hypothesis_offset,
    ) + _hirschberg_alignment(
        reference[midpoint:],
        hypothesis[split:],
        reference_offset + midpoint,
        hypothesis_offset + split,
    )
