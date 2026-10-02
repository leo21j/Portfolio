"""Raw-to-clean dataset construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from complaint_triage.io import dataset_profile, read_complaints_csv, write_csv
from complaint_triage.preprocessing import prepare_complaints
from complaint_triage.utils.config import save_json


def build_clean_dataset(
    input_path: str | Path,
    output_path: str | Path,
    profile_output_path: str | Path | None = None,
    min_text_chars: int = 40,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read raw complaints, filter valid narratives, and save clean data."""
    raw = read_complaints_csv(input_path, nrows=nrows)
    clean = prepare_complaints(raw, min_text_chars=min_text_chars)
    write_csv(clean, output_path)

    if profile_output_path is not None:
        save_json(dataset_profile(clean), profile_output_path)

    return clean
