# Jinder

Upload a resume, get a ranked job and grant matches.

Jinder pulls listings from three public sources, indexes them, and ranks them
against the full text of a résumé using hybrid retrieval — dense vector search
combined with BM25 keyword search, rather than either one alone. It runs as a
FastAPI service on AWS with a scheduled Lambda keeping the listings fresh.


## Why hybrid retrieval

The interesting problem here is ranking, not plumbing.

Dense vector search understands that "ML engineer" and "machine learning
developer" are the same job, but it blurs rare, precise tokens — a specific
grant program name, a clearance level, a framework version. BM25 is the
opposite: exact on rare terms, blind to paraphrase. Résumés and job postings
need both, because a résumé is a long paraphrase-heavy document that also
contains exactly the kind of specific keywords an employer filters on.

Jinder runs both retrievers and fuses their scores:

1. **Normalize separately.** BM25 returns unbounded term-frequency sums; the
   vector index returns cosine similarities in a narrow band. Each list is
   standardized to a z-score, then squashed logistically into 0–1. A z-score is
   used instead of min-max because min-max forces the worst result in every
   batch to exactly zero, which throws away the difference between "weakest of a
   strong set" and "genuinely irrelevant."
2. **Blend with a weight.** A single `alpha` parameter trades the two off —
   `1.0` is vector-only, `0.0` is BM25-only, default `0.5`.
3. **Reward agreement.** A listing found by *both* retrievers accumulates both
   contributions, so consensus outranks a strong hit from either side alone.
   A listing found by only one keeps just that side's share.

Two optional stages sit on top:

- **Graph expansion** pulls in listings adjacent to strong hits in an embedding
  neighborhood graph, to surface near-misses the retrievers ranked just below
  the cutoff.
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`) re-scores the top
  candidates with full query-document attention. Off by default: it is
  materially slower and only worth it for small `k`.

The fusion logic is in [`services/job_matcher/src/scoring.py`](services/job_matcher/src/scoring.py),
deliberately free of model imports so it can be unit tested without downloading
anything. [`tests/test_scoring.py`](tests/test_scoring.py) covers the scale
invariance, the alpha endpoints, and the consensus property described above.

## Architecture

<p align="center">
  <img src="assets/ingestion.png" alt="Ingestion architecture" width="700">
</p>

| Component | Role |
|---|---|
| **Ingestion Lambda** (`lambda_function.py`) | Runs on a schedule. Fetches from JSearch, USAJOBS, and Grants.gov; writes raw JSON to S3; calls `POST /reload` on the API. |
| **S3** | Raw listing dumps, keyed `raw_data/<source>_<timestamp>.json`. The API reads the newest dump per source. |
| **FastAPI service** (`services/api.py`) | Loads listings at startup, builds the indices, serves the frontend and the matching endpoint. |
| **Job matcher** (`services/job_matcher/`) | Résumé parsing, embedding, and the hybrid retrieval stack. |
| **User service** (`services/user_service/`) | Accounts and preferences in DynamoDB. Salted PBKDF2 password hashing. |
| **Frontend** (`frontend/`) | Vanilla JS single-page app. No build step. |

One design note worth calling out: ingestion is decoupled from serving through
S3 rather than wired directly. A failing API deploy cannot lose a scheduled
fetch, a rate-limited source does not block the others, and the raw dumps stay
replayable. `fetch_and_upload` returns a per-source status map so a partial run
reports as `partial` instead of looking clean.

### Data sources

| Source | Auth | Notes |
|---|---|---|
| [JSearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch) (RapidAPI) | API key | Aggregated job postings |
| [USAJOBS](https://developer.usajobs.gov/api-reference/get-api-search) | API key + registered email | US federal positions |
| [Grants.gov](https://www.grants.gov/api/api-guide) | None | Federal grant opportunities |

Request and response shapes for all three are documented in
[`docs/APIS.md`](docs/APIS.md).

## Setup

Requires Python 3.10+, an AWS account, and keys for JSearch and USAJOBS.

```bash
git clone https://github.com/leo21j/Jinder.git
cd Jinder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# PyTorch is installed separately; the right wheel depends on your hardware.
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Configure credentials:

```bash
cp services/.env.example services/.env
# then fill in the values
```

| Variable | Required | Purpose |
|---|---|---|
| `S3_BUCKET_NAME` | yes | Bucket for raw listing dumps |
| `S3_KEY_PREFIX` | no | Key prefix, defaults to `raw_data/` |
| `JINDER_USERS_TABLE` | no | DynamoDB table, defaults to `JinderUsers` |
| `AWS_REGION` | no | Defaults to `us-east-1` |
| `RAPIDAPI_KEY` | yes | JSearch |
| `USAJOBS_API_KEY` | yes | USAJOBS |
| `USAJOBS_EMAIL` | yes | Sent as `User-Agent`; USAJOBS requires it |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins, defaults to `http://localhost:8000` |
| `FASTAPI_URL` | no | Lambda only — base URL to call `/reload` on |

AWS prerequisites: an S3 bucket, a DynamoDB table keyed on `email` (string),
and credentials in the environment or `~/.aws/credentials` with read/write on
both.

Run it:

```bash
uvicorn services.api:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. First startup downloads the embedding model
and, if the bucket is empty, fetches from all three sources — expect a few
minutes. `GET /healthcheck` reports how many listings loaded.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Frontend |
| `/healthcheck` | GET | Status and loaded listing count |
| `/match` | POST | Rank listings against an uploaded résumé PDF (`file`, `top_k`) |
| `/reload` | POST | Re-read S3 and rebuild the indices |
| `/register` | POST | Create an account |
| `/login` | POST | Verify credentials |
| `/profile` | GET / POST | Read or update preferences |

```bash
curl -X POST http://localhost:8000/match \
  -F "file=@resume.pdf" -F "top_k=5"
```

Interactive docs at `/docs`.

## Development

```bash
pip install ruff pytest
ruff check .
pytest
```

The test suite covers score normalization, score fusion, and password hashing —
all of which run without the embedding model, so it is fast and offline. CI runs
lint, tests, and a check that no `.env` file is ever tracked.

For an end-to-end check against the committed sample data (needs the full
dependency set):

```bash
python scripts/check_job_matcher.py
```

## Repository structure

```text
.
├── lambda_function.py          # Scheduled ingestion Lambda
├── services/
│   ├── api.py                  # FastAPI app: routes, startup, CORS
│   ├── ingestion.py            # Shared fetch-and-upload, used by API and Lambda
│   ├── job_fetcher/            # JSearch, USAJOBS, Grants.gov clients
│   ├── job_matcher/
│   │   ├── job_matcher.py      # Résumé parsing, embedding, orchestration
│   │   └── src/
│   │       ├── scoring.py      # Score normalization and fusion (pure, tested)
│   │       ├── hybrid_retriever.py
│   │       ├── vector_retriever.py
│   │       ├── bm25_retriever.py
│   │       └── graph_retriever.py
│   └── user_service/           # DynamoDB accounts, PBKDF2 hashing
├── frontend/                   # Vanilla JS SPA, no build step
├── scripts/                    # Manual end-to-end check
├── sample_data/                # Synthetic examples for offline runs
├── docs/APIS.md                # Source API reference
├── design.md                   # Component responsibilities and data model
└── tests/
```

## Built with

**Backend:** Python 3.10, FastAPI, uvicorn, Pydantic
**Retrieval:** sentence-transformers, FAISS, rank-bm25, PyTorch, NumPy
**Cloud:** AWS Lambda, S3, DynamoDB, EC2
**Parsing:** PyMuPDF
**Frontend:** vanilla JavaScript, no framework
**Tooling:** pytest, ruff, GitHub Actions


## Possible next steps

- Build a small labeled relevance set and actually tune `alpha`
- Weight résumé sections, and recency, instead of using flat full text
- Real session auth, and move the admin gate server-side
- Incremental index updates rather than full rebuilds
- Cache embeddings in S3 so restarts do not re-embed from scratch

## Credits

Originally a three-person graduate cloud-computing course project by
Leo J Robles and two colaborators. This is a joint work