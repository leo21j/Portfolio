"""Text matching helpers shared by the triage and escalation rule sets."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def to_lower_text(value: Any) -> str:
    """Coerce any cell value to lowercase text, mapping nulls to an empty string."""
    if pd.isna(value):
        return ""
    return str(value).lower()


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Check whether ``haystack`` contains ``phrase`` at a word start.

    The match is anchored at the beginning of a word but left open at the end,
    which is the behavior a keyword list wants: "charge" should fire on
    "charged" and "charges", while "sue" must not fire on "issue".

    That asymmetry matters. An unanchored substring search makes the keyword
    "sue" match 17% of CFPB narratives via the word "issue" and pushes almost
    every row toward the legal category. Anchoring both ends instead would be
    safe but would miss ordinary inflections.

    Phrases that begin with a non-word character fall back to a plain
    substring check, since the leading assertion cannot anchor to punctuation.
    """
    phrase = phrase.strip().lower()
    if not phrase:
        return False

    if not phrase[0].isalnum():
        return phrase in haystack

    return re.search(rf"(?<!\w){re.escape(phrase)}", haystack) is not None


def join_fields(row: pd.Series, columns: list[str]) -> str:
    """Concatenate the named row fields into one lowercase string for matching."""
    return " | ".join(to_lower_text(row.get(column, "")) for column in columns)
