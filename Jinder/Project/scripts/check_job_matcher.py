"""Manual end-to-end check of the matching pipeline against the sample data.

Not a unit test -- it loads the real embedding model and needs the full
dependency set, so it is kept out of ``tests/``. Run it from the repo root
after a dependency change to confirm the pipeline still produces sensible
rankings:

    python scripts/check_job_matcher.py

The automated tests covering score normalization and fusion live in
``tests/test_scoring.py`` and need no model downloads.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Must be set before faiss and torch are imported; see services/api.py.
if sys.platform == "darwin":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, "1")

from services.job_matcher.job_matcher import JobMatcher  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_FILES = [
    "sample_data/jsearch_example.json",
    "sample_data/usajobs_example.json",
    "sample_data/grants_example.json",
]
SAMPLE_RESUME = "sample_data/tommy_trojan_resume.pdf"
TOP_K = 5


def main() -> int:
    matcher = JobMatcher()

    loaded = 0
    for relative in SAMPLE_FILES:
        path = REPO_ROOT / relative
        if not path.exists():
            logger.warning("Missing sample file: %s", relative)
            continue
        logger.info("Loading jobs from %s", relative)
        matcher.load_jobs_from_json(str(path))
        loaded += 1

    if not loaded:
        logger.error("No sample data found. Run from the repo root.")
        return 1

    matcher.generate_embeddings()

    resume = REPO_ROOT / SAMPLE_RESUME
    if not resume.exists():
        logger.error("Sample résumé not found: %s", SAMPLE_RESUME)
        return 1

    resume_info = matcher.parse_from_resume(str(resume))
    logger.info("Résumé text length: %s characters", len(resume_info["full_text"]))

    matches = matcher.match_job(resume_info["full_text"], top_k=TOP_K)
    if not matches:
        logger.error("No matches returned, which suggests a broken index.")
        return 1

    print(f"\nTop {len(matches)} matches:")
    for rank, match in enumerate(matches, start=1):
        print(
            f"{rank}. [{float(match.get('match_score', 0.0)):.3f}] "
            f"{match.get('title') or '<no title>'} - "
            f"{match.get('company') or '<no company>'} "
            f"({match.get('location') or '<no location>'})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
