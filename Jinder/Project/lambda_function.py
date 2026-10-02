"""Scheduled ingestion Lambda.

Fetches job and grant listings from the source APIs, uploads the raw JSON to
S3, then asks the FastAPI service to reload so the new data is searchable.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import boto3
import requests

from services import ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base URL of the running FastAPI service. Leave unset to skip the reload call.
FASTAPI_URL = os.environ.get("FASTAPI_URL")


def handler(event, context):
    """Fetch every source, upload to S3, and trigger a reload."""
    bucket = ingestion.get_bucket()

    results = ingestion.fetch_and_upload(s3=boto3.client("s3"), bucket=bucket)
    uploaded = [source for source, state in results.items() if state == "uploaded"]
    logger.info("Uploaded %s of %s sources to S3", len(uploaded), len(results))

    if FASTAPI_URL and uploaded:
        try:
            requests.post(f"{FASTAPI_URL}/reload", timeout=30)
            logger.info("Triggered reload at %s", FASTAPI_URL)
        except requests.RequestException as exc:
            logger.error("Failed to call /reload: %s", exc)
            results["reload"] = "failed"

    # "success" only when every source made it to S3, so a partial run is
    # visible in the Lambda console instead of looking clean.
    status = "success" if len(uploaded) == len(ingestion.SOURCES) else "partial"
    if not uploaded:
        status = "failed"

    return {"status": status, "sources": results}


if __name__ == "__main__":
    try:
        logger.info(json.dumps(handler({}, None), indent=2))
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
