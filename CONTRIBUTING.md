# Contributing

Thank you for your interest in `ai-mutation`. This project follows the
JOSS open-source contribution model.

## Getting set up

```bash
git clone https://github.com/<owner>/ai-mutation
cd ai-mutation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov ruff
cp .env.example .env   # then set AI_BASE_URL, AI_MODEL, optional AI_API_KEY, NCBI_EMAIL, ...
```

## Running the test suite

```bash
pytest -q
ruff check src eval tests --select E,F,W,I --ignore E501
```

All tests are offline and deterministic — no LLM call, no network call.
CI runs them on Python 3.10, 3.11, and 3.12.

## Reporting issues

When opening an issue, please include:

* the exact mutation query that triggered the problem;
* the model name (`AI_MODEL`);
* the `run` metadata block from the relevant `ReasoningResult` (it
  contains the prompt hash and seed);
* whether `deterministic=True` was used.

## Pull-request checklist

1. New behaviour is covered by a test.
2. `pytest -q` and `ruff check …` are green locally.
3. Public-facing changes are reflected in `README.md`.
4. If you touch `src/verification.py`, you must add a regression test in
   `tests/test_verification.py` — the verifier is the methodological
   core of the project and is exercised by the evaluation harness.
5. By submitting a PR you agree to license your contribution under the
   project's MIT license.

## Code style

* `from __future__ import annotations` in every module.
* Functions that touch external services live in `src/data_sources.py`
  or `src/structure.py`; pure logic lives elsewhere and must be
  unit-testable without the network.
* No new dependencies without justification — every package adds
  install friction for downstream users.

## Disclosure

This is a research-use-only tool. Do not submit issues or pull requests
that imply clinical use without an accompanying disclaimer.
