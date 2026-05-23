# MLSys 2026 Track B — competition-mirror sandbox
# Replicates organizer environment: Python 3.12, no network except Gemini API
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy source
COPY agent.py orchestrator.py ./
COPY agents/ agents/
COPY core/ core/
COPY prompts/ prompts/

# Data and output directories (mount these at runtime)
RUN mkdir -p /data /output

# GOOGLE_API_KEY is injected at runtime via --env or docker-compose
ENV PYTHONUNBUFFERED=1

# Usage: docker run ... agent input.json output.json
# (input/output are mounted paths — see docker-compose.yml)
ENTRYPOINT ["python3", "agent.py"]
