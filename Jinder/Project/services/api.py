"""FastAPI service for Jinder.

Serves the frontend, exposes résumé matching, and handles user accounts. Job
data is loaded from S3 at startup and can be refreshed via POST /reload, which
the ingestion Lambda calls after it uploads new listings.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from services import ingestion
from services.job_matcher.job_matcher import JobMatcher
from services.user_service.user_service import UserService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Single-threading the BLAS backends works around a segfault when faiss and
# PyTorch are loaded in the same process on macOS. It costs throughput, so it
# is limited to the platform that needs it.
if sys.platform == "darwin":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, "1")

# Comma-separated list of allowed origins, for example
# "https://jinder.example.com,http://localhost:8000". Defaults to localhost so
# a misconfigured deployment fails closed instead of allowing every origin.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

s3 = boto3.client("s3")
user_service = UserService()
job_matcher: JobMatcher | None = None


def _load_jobs_into(matcher: JobMatcher) -> None:
    """Load the newest dump for every source from S3 into the matcher."""
    bucket = ingestion.get_bucket()
    for source in ingestion.SOURCES:
        key = ingestion.latest_key(s3, bucket, source)
        if key:
            matcher.load_jobs_from_dict(ingestion.load_json(s3, bucket, key), source=source)
            logger.info("Loaded %s", key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Populate the matcher before the service accepts traffic."""
    global job_matcher
    matcher = JobMatcher()

    bucket = ingestion.get_bucket()
    if ingestion.is_bucket_empty(s3, bucket):
        logger.info("No data in S3, fetching from the source APIs...")
        ingestion.fetch_and_upload(s3, bucket)

    _load_jobs_into(matcher)
    matcher.generate_embeddings()
    job_matcher = matcher

    logger.info("Startup complete. %s jobs loaded", len(matcher.jobs_metadata))
    yield


app = FastAPI(title="Jinder API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "ok",
        "jobs_loaded": len(job_matcher.jobs_metadata) if job_matcher else 0,
    }


@app.post("/reload")
async def reload():
    """Reload job data from S3. Called by the ingestion Lambda after upload."""
    global job_matcher
    logger.info("Reloading jobs from S3...")

    matcher = JobMatcher()
    _load_jobs_into(matcher)
    matcher.generate_embeddings()
    job_matcher = matcher

    logger.info("Reload complete. %s jobs loaded", len(matcher.jobs_metadata))
    return {"status": "reloaded", "jobs_count": len(matcher.jobs_metadata)}


@app.post("/match")
async def match_job(file: UploadFile = File(...), top_k: int = 10):
    """Rank jobs and grants against an uploaded résumé PDF."""
    if job_matcher is None:
        raise HTTPException(status_code=503, detail="Job Matcher not initialized.")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        resume = job_matcher.parse_from_resume(tmp_path)
        match_results = job_matcher.match_job(resume["full_text"], top_k=top_k)
    except Exception as exc:
        logger.exception("Matching failed")
        raise HTTPException(status_code=500, detail=f"Failed to match job: {exc}") from exc
    finally:
        # NamedTemporaryFile(delete=False) leaves the file behind, so every
        # upload would otherwise leak a résumé into the temp directory.
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {"match_results": match_results}


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    default_title: str | None = None
    default_location: str | None = None
    remote_pref: str | None = None


class UpdateProfileRequest(BaseModel):
    email: str
    default_title: str | None = None
    default_location: str | None = None
    remote_pref: str | None = None


@app.post("/login")
async def login(body: LoginRequest):
    email = body.email.strip()
    if not email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    user = user_service.check_credentials(email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"user": user}


@app.post("/register")
async def register(body: RegisterRequest):
    email = body.email.strip().lower()
    if not email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    try:
        user = user_service.create_user(
            email=email,
            password=body.password,
            default_title=body.default_title,
            default_location=body.default_location,
            remote_pref=body.remote_pref,
        )
    except ValueError as exc:
        # Raised when the email is already registered.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Registration failed")
        raise HTTPException(status_code=500, detail="Failed to create user.") from exc
    return {"user": user}


@app.get("/profile")
async def get_profile(email: str):
    user = user_service.get_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.pop("password_hash", None)
    return {"user": user}


@app.post("/profile")
async def update_profile(body: UpdateProfileRequest):
    updated_user = user_service.update_preferences(
        email=body.email,
        default_title=body.default_title,
        default_location=body.default_location,
        remote_pref=body.remote_pref,
    )
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found.")
    updated_user.pop("password_hash", None)
    return {"user": updated_user}
