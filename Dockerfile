# MLSys 2026 Track B — competition-mirror sandbox
# Replicates organizer environment: Python 3.12, no network except Gemini API
FROM python:3.12-slim

WORKDIR /app

# Copy package metadata first for dependency-only layer cache.
# Install only declared dependencies (no source yet) so this layer reuses across edits.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir \
      google-genai>=1.0.0 \
      networkx>=3.6.1 \
      pydantic>=2.12.5 \
      python-dotenv>=1.0.0 \
      rich>=14.3.3

# Copy source — keeps install logic separate from edit cache invalidation
COPY agent.py orchestrator.py ./
COPY agents/ agents/
COPY core/ core/
COPY prompts/ prompts/

# Install the package itself (uses already-cached deps)
RUN pip install --no-cache-dir --no-deps .

# Data and output directories (mount these at runtime)
RUN mkdir -p /data /output

# GOOGLE_API_KEY is injected at runtime via --env or docker-compose
ENV PYTHONUNBUFFERED=1

# Usage: docker run ... agent input.json output.json
# (input/output are mounted paths — see docker-compose.yml)
ENTRYPOINT ["python3", "agent.py"]
