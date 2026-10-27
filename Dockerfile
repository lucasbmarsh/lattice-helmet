# Lattice service: FastAPI app + engine. Listens on $PORT (default 8080).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LATTICE_WORKDIR=/tmp/jobs PORT=8080
WORKDIR /srv

COPY pyproject.toml README.md ./
COPY lattice_helmet lattice_helmet
RUN pip install '.[app,decimate]'
COPY app app
COPY recipes recipes

# --proxy-headers so redirect URIs come out as https behind the host's proxy.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips '*'
