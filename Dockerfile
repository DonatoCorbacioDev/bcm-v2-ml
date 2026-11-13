# 3.12 is the oldest Python numpy 2.5 supports. Before any bump, check that
# every pinned package has a prebuilt wheel for it on both amd64 and arm64
# (the VM): python:3.x-slim has no C compiler, so a missing wheel fails the
# build (meson.build: "Unknown compiler(s)") instead of compiling from source.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user for security. HOME is set to /app (rather than a
# separate home dir) because some dependencies (e.g. Prophet's Stan
# backend, matplotlib) write cache files under $HOME at runtime.
RUN groupadd -r ml && useradd -r -g ml -d /app ml && chown -R ml:ml /app
USER ml
ENV HOME=/app

EXPOSE 8000

CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
