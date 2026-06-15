# FactorForge — offline demo image (mock LLM + mock agents + synthetic data; no credentials).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install dependencies first (better layer caching), then the package itself.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
COPY evals ./evals
RUN pip install -e "."

ENTRYPOINT ["factorforge"]
CMD ["demo"]
