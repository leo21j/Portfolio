import json
import os
from pathlib import Path
from typing import Any

import requests

"""
 I just muted all the save to json call as im doing some local testing. - Ray
"""

# folder where we’ll store raw JSON snapshots
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
# SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
# SAMPLE_DATA_DIR.mkdir(exist_ok=True)


# save any JSON to a file

def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# JSEARCH FETCHER

def fetch_jsearch(
    title: str | None = None,
    location: str | None = None,
    remote: bool | None = None,
    date_posted: str = "week",
    page: int = 1,
    num_pages: int = 1,
) -> dict[str, Any]:
    """
    call JSearch via RapidAPI and return the raw JSON response.
    Also saves the result to sample_data/jsearch_live.json.
    """
    api_key = os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        raise RuntimeError("RAPIDAPI_KEY env var is not set")

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com",
    }

    # query text
    query_parts = []
    if title:
        query_parts.append(title)
    if location:
        query_parts.append(f"in {location}")

    # if nothing specified
    if query_parts:
        query = " ".join(query_parts)
    else:
        query = "jobs"

    params: dict[str, Any] = {
        "query": query,
        "page": page,
        "num_pages": num_pages,
        "date_posted": date_posted,  #
    }

    if remote is not None:
        # API expects string "true"/"false"
        params["remote_jobs_only"] = "true" if remote else "false"

    resp = requests.get(
        "https://jsearch.p.rapidapi.com/search",
        headers=headers,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()

    return raw


#  USAJOBS FETCHER

def fetch_usajobs(
    title: str | None = None,
    location: str | None = None,
    remote: bool | None = None,
    date_posted_days: int = 7,
    results_per_page: int = 25,
    page: int = 1,
) -> dict[str, Any]:
    """
    call USAJobs search endpoint and return raw JSON
    """
    email = os.environ.get("USAJOBS_EMAIL")
    api_key = os.environ.get("USAJOBS_API_KEY")
    if not email or not api_key:
        raise RuntimeError("USAJOBS_EMAIL or USAJOBS_API_KEY env vars not set")

    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": email,
        "Authorization-Key": api_key,
    }

    # keyword if no specific title
    keyword = title or "jobs"

    params: dict[str, Any] = {
        "Keyword": keyword,
        "ResultsPerPage": results_per_page,
        "Page": page,
        "DatePosted": max(0, min(date_posted_days, 60)),  # clamp 0–60
    }

    if location:
        params["LocationName"] = location

    if remote is not None:
        params["RemoteIndicator"] = "True" if remote else "False"

    resp = requests.get(
        "https://data.usajobs.gov/api/search",
        headers=headers,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()  # full USAJobs

    return raw


#  GRANTS.GOV FETCHER

def fetch_grants(
    keyword: str | None = None,
    rows: int = 10,
    opp_statuses: str = "forecasted|posted",
) -> dict[str, Any]:
    """
    Call Grants.gov search2 endpoint and return raw JSON.
    Also saves to sample_data/grants_live.json.
    """
    url = "https://api.grants.gov/v1/api/search2"

    effective_keyword = keyword or "education"

    body: dict[str, Any] = {
        "keyword": effective_keyword,
        "rows": rows,
        "oppStatuses": opp_statuses,
        # Add 'eligibilities', 'agencies', etc. if you want later
    }

    headers = {
        "Content-Type": "application/json",
    }

    resp = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()

    # The matcher can decide what to do with it
    return raw


#  inified entrypoint (for Flask / Lambda)

def lambda_handler(event, context):
    """
    Unified fetcher for Jinder
    event can contain:
      - title:
      - location: optional
      - remote: "Yes" , "No"  , None
      - days: (default 7)
      - jsearch_date_posted:
      - grants_keyword:
      - grants_rows: how many grants to fetch (default 10)

    Returns raw JSON from each source plus simple stats.
    """
    payload = event or {}

    title = payload.get("title")          # can be None
    location = payload.get("location")    # can be None
    remote_pref = payload.get("remote")   # Yes/No/None

    days = int(payload.get("days") or 7)
    jsearch_date = payload.get("jsearch_date_posted") or "week"
    grants_keyword = payload.get("grants_keyword") or title or "education"
    grants_rows = int(payload.get("grants_rows") or 10)

    # remote flag  Yes/No/None -> True/False/None
    remote_bool: bool | None
    if isinstance(remote_pref, str):
        if remote_pref.lower() == "yes":
            remote_bool = True
        elif remote_pref.lower() == "no":
            remote_bool = False
        else:
            remote_bool = None
    else:
        remote_bool = None

    results: dict[str, Any] = {
        "jsearch_raw": None,
        "usajobs_raw": None,
        "grants_raw": None,
        "source_errors": {},
    }

    # JSearch
    try:
        jsearch_raw = fetch_jsearch(
            title=title,
            location=location,
            remote=remote_bool,
            date_posted=jsearch_date,
        )
        results["jsearch_raw"] = jsearch_raw
    except Exception as e:
        results["source_errors"]["jsearch"] = str(e)

    # USAJobs
    try:
        usajobs_raw = fetch_usajobs(
            title=title,
            location=location,
            remote=remote_bool,
            date_posted_days=days,
            results_per_page=25,
        )
        results["usajobs_raw"] = usajobs_raw
    except Exception as e:
        results["source_errors"]["usajobs"] = str(e)

    # Grants.gov
    try:
        grants_raw = fetch_grants(
            keyword=grants_keyword,
            rows=grants_rows,
        )
        results["grants_raw"] = grants_raw
    except Exception as e:
        results["source_errors"]["grants"] = str(e)

    return results
