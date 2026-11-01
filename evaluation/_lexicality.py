"""Tokenize GoBo outputs and classify them against the fixed SCOWL list."""

import unicodedata
from pathlib import Path
from typing import Callable

import regex as re

TOKEN_RE = re.compile(r"[\p{L}\p{M}]+(?:[-'’][\p{L}\p{M}]+)*")
WORD_FORM_RE = re.compile(r"^[\p{L}\p{M}]+(?:[-'’][\p{L}\p{M}]+)*$")
ORDINARY_BOUNDARY_RE = re.compile(r"[,.;:!()\[\]{}—–]+")
EDGE_PUNCTUATION_RE = re.compile(r"^[\p{P}\p{S}]+|[\p{P}\p{S}]+$")
ALLOWED_NONWORD_RE = re.compile(r"""^[\s.,!?;:'"“”‘’()\[\]{}—–-]*$""")
ONE_LETTER_ENGLISH_WORDS = {"a", "i"}


def normalize_token(token: str) -> str:
    return unicodedata.normalize("NFC", token).replace("’", "'").lower()


def tokenize_with_rejections(text: str) -> tuple[list[str], list[str]]:
    normalized = unicodedata.normalize("NFC", text)
    tokens = []
    rejected = []
    for chunk in ORDINARY_BOUNDARY_RE.sub(" ", normalized).split():
        candidate = EDGE_PUNCTUATION_RE.sub("", chunk)
        if candidate and WORD_FORM_RE.fullmatch(candidate):
            tokens.append(normalize_token(candidate))
        elif candidate:
            rejected.append(chunk)
    return tokens, rejected


def tokenize(text: str) -> list[str]:
    return tokenize_with_rejections(text)[0]


def load_scowl(path: Path) -> set[str]:
    words: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()

            if raw and (raw == raw.lower() or raw == "I"):
                words.add(normalize_token(raw))
    return words


def classify(
    text: str,
    is_scowl: Callable[[str], bool],
) -> dict:
    normalized_text = unicodedata.normalize("NFC", text)
    text_tokens, rejected_chunks = tokenize_with_rejections(normalized_text)
    residue = TOKEN_RE.sub("", normalized_text)
    lexical_line_form = not rejected_chunks and bool(
        ALLOWED_NONWORD_RE.fullmatch(residue)
    )
    known_count = sum(
        (len(token) > 1 or token in ONE_LETTER_ENGLISH_WORDS) and is_scowl(token)
        for token in text_tokens
    )
    token_count = len(text_tokens)
    return {
        "token_count": token_count,
        "scowl_known_count": known_count,
        "scowl_contains_known": known_count > 0,
        "scowl_all_known": lexical_line_form
        and token_count > 0
        and known_count == token_count,
    }


def aggregate(rows: list[dict]) -> dict:
    samples = len(rows)
    total_tokens = sum(row["token_count"] for row in rows)
    known = sum(row["scowl_known_count"] for row in rows)
    lines_with = sum(row["scowl_contains_known"] for row in rows)
    lines_all = sum(row["scowl_all_known"] for row in rows)
    return {
        "samples": samples,
        "tokens": total_tokens,
        "known_tokens": known,
        "lines_with_at_least_one_known": lines_with,
        "all_known_lines": lines_all,
    }
