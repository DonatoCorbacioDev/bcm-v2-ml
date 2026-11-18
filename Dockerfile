# 3.12 is the oldest Python numpy 2.5 supports. Before any bump, check that
# every pinned package has a prebuilt wheel for it on both amd64 and arm64
# (the VM): python:3.x-slim has no C compiler, so a missing wheel fails the
# build (meson.build: "Unknown compiler(s)") instead of compiling from source.
FROM python:3.12-slim

# The base image's Debian packages brought up to date: it is rebuilt less
# often than Debian publishes fixes (zlib, Trivy, 2026-11-18).
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bytecode compiled now, as root: the code stays owned by root and read-only
# to the service (OWASP Docker Security, non-root user), which therefore
# cannot rewrite it, and nothing is written next to it at runtime.
RUN python -m compileall -q /app/app

# A non-root user. Its HOME is /tmp, the only writable place (a tmpfs in
# docker-compose.yml, the root filesystem being read-only): matplotlib's
# cache, Prophet's Stan output and gunicorn's control socket go there.
RUN groupadd -r ml && useradd -r -g ml -d /tmp ml
USER ml
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
