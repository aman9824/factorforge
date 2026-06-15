.PHONY: install install-vertex install-claude install-live install-french \
        test lint typecheck check demo run eval serve-mcp fetch-french api docker clean

PY ?= python
Q ?= Does the value premium depend on the inflation regime?

# ffn (pulled in by bt) imports matplotlib.pyplot at import time; force a headless backend so
# every recipe is deterministic in CI / on a server with no display.
export MPLBACKEND := Agg

install:           ## Install package + dev deps (offline mock path — no credentials needed)
	$(PY) -m pip install -e ".[dev]"

install-vertex:    ## + AnthropicVertex (real structured generation)
	$(PY) -m pip install -e ".[dev,vertex]"

install-claude:    ## + Claude Agent SDK (real multi-agent pipeline)
	$(PY) -m pip install -e ".[dev,claude]"

install-live:      ## + both real LLM seams (Vertex extraction + Claude agents)
	$(PY) -m pip install -e ".[dev,vertex,claude]"

install-french:    ## + pandas-datareader (fetch real Fama-French data locally)
	$(PY) -m pip install -e ".[dev,french]"

test:              ## Run the test suite
	$(PY) -m pytest

lint:              ## Lint with ruff
	$(PY) -m ruff check .

typecheck:         ## Type-check with mypy (strict)
	$(PY) -m mypy

check: lint typecheck test eval  ## Everything CI runs

demo:              ## End-to-end research run on the offline mock + synthetic data
	$(PY) -m factorforge.cli demo

run:               ## Research your own question: make run Q="..."
	$(PY) -m factorforge.cli research "$(Q)"

eval:              ## Run the eval suite and gate on thresholds
	$(PY) -m evals.run_evals

serve-mcp:         ## Run the standalone knowledge-graph MCP server over streamable-HTTP
	$(PY) -m factorforge.mcp_server --http

fetch-french:      ## Fetch real Fama-French factor data into the git-ignored cache (needs [french])
	$(PY) -m factorforge.cli fetch-french

api:               ## Serve the HTTP API on :8000
	$(PY) -m uvicorn factorforge.api:app --host 0.0.0.0 --port 8000

docker:            ## Build the image (also works with: podman build -t factorforge .)
	docker build -t factorforge .

clean:             ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info out
