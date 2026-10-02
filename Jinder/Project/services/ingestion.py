"""Shared job and grant ingestion.

Fetches raw listings from the three source APIs and uploads them to S3. Both the
scheduled Lambda (``lambda_function.py``) and the FastAPI service
(``services/api.py``) call into here, so the two cannot drift apart.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from services.job_fetcher.job_fetcher import fetch_grants, fetch_jsearch, fetch_usajobs

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "raw_data/"

# Source name -> fetch function. The name becomes the S3 key prefix, so
# `jsearch` lands at `<prefix>jsearch_<timestamp>.json`.
SOURCES: dict[str, Callable[[], dict[str, Any]]] = {
    "jsearch": fetch_jsearch,
    "usajobs": fetch_usajobs,
    "grants": fetch_grants,
}


def get_bucket() -> str:
    """Return the configured S3 bucket, or raise if it is not set."""
    bucket = os.environ.get("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError(
            "S3_BUCKET_NAME is not set. Copy services/.env.example to "
            "services/.env and fill it in."
        )
    return bucket


def get_prefix() -> str:
    return os.environ.get("S3_KEY_PREFIX", DEFAULT_PREFIX)


def upload_json(s3, bucket: str, key: str, data: Any) -> None:
    """Upload a JSON-serializable payload to S3."""
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json",
    )
    logger.info("Uploaded s3://%s/%s", bucket, key)


def latest_key(s3, bucket: str, source: str, prefix: str | None = None) -> str | None:
    """Return the most recently modified S3 key for one source, if any."""
    prefix = prefix if prefix is not None else get_prefix()
    response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}{source}_")
    contents = response.get("Contents")
    if not contents:
        return None
    newest = max(contents, key=lambda item: item["LastModified"])
    return newest["Key"]


def load_json(s3, bucket: str, key: str) -> Any:
    """Read and parse a JSON object from S3."""
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def is_bucket_empty(s3, bucket: str) -> bool:
    """True when no source has any uploaded data yet."""
    return all(latest_key(s3, bucket, source) is None for source in SOURCES)


def fetch_and_upload(s3, bucket: str, prefix: str | None = None) -> dict[str, str]:
    """Fetch every source and upload whatever succeeded.

    One failing source does not stop the others, since a stale JSearch key
    should not block the free Grants.gov feed. Returns a per-source status map
    so callers can decide what to report.
    """
    prefix = prefix if prefix is not None else get_prefix()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results: dict[str, str] = {}

    for source, fetch in SOURCES.items():
        try:
            logger.info("Fetching %s...", source)
            data = fetch()
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", source, exc)
            results[source] = "fetch_failed"
            continue

        if not data:
            logger.warning("No %s data returned", source)
            results[source] = "empty"
            continue

        try:
            upload_json(s3, bucket, f"{prefix}{source}_{timestamp}.json", data)
            results[source] = "uploaded"
        except Exception as exc:
            logger.error("Failed to upload %s: %s", source, exc)
            results[source] = "upload_failed"

    return results
