# CPU-only image for running the data and baseline stages. Transformer training
# needs a CUDA base image and a CUDA-enabled torch wheel; see docs/gpu_setup.md.
FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

# Mount the data directory at run time, for example:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" <image> \
#     python scripts/01_prepare_data.py --input data/raw/complaints.csv \
#       --output data/interim/complaints_clean.csv
CMD ["python", "-c", "import complaint_triage; print('complaint_triage', complaint_triage.__version__)"]
