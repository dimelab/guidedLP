# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

The Python package is nested one level deep. The actual project root for packaging and tooling is `guidedLP/`; the repo root just holds a smoke test and the high-level README.

```
guidedLabelPropagation/           <- repo root
├── test_installation.py          <- post-install smoke test (run from here)
├── README.md, requirements.txt
└── guidedLP/                     <- Python project root (install from here)
    ├── setup.py                  <- single canonical install config
    ├── pytest.ini, .ruff.toml, .mypy.ini, .pre-commit-config.yaml, .coveragerc
    ├── src/guidedLP/             <- actual package source
    │   ├── common/    (IDMapper, exceptions, validators, logging_config)
    │   ├── network/   (construction, analysis, communities, backboning, filtering, export)
    │   ├── glp/       (propagation, validation, evaluation, utils)
    │   └── timeseries/(slicing, temporal_metrics, category_analysis)
    ├── tests/                    <- mirrors src/guidedLP/ layout + test_integration.py
    ├── docs/                     <- architecture/, api/, specifications/
    ├── examples/
    └── guidedLP_OLD/             <- reference implementation (net_utils.py / stlp())
```

The install is configured via plain `setup.py` (no `pyproject.toml`) — `setup.py` lives at `guidedLP/setup.py`, uses `package_dir={"": "src"}` + `find_packages(where="src")`, and exposes the package as importable `guidedLP`.

## Common Commands

Run from `guidedLP/` unless noted.

```bash
# Install in dev mode
pip install -e .
pip install -e ".[dev,docs,viz]"             # with optional extras

# Post-install smoke test (run from repo root)
python test_installation.py

# Test suite
pytest                                       # all tests
pytest tests/glp/test_propagation.py         # single file
pytest tests/glp/test_propagation.py::TestClass::test_method
pytest -m "fast and not external"            # by marker
pytest -m "glp and small_data"
pytest -n auto                               # parallel (pytest-xdist)
pytest --cov=src --cov-report=html           # with coverage

# Lint / format / types
ruff check src/ tests/ [--fix]
black src/ tests/
mypy src/

# Pre-commit
pre-commit install
pre-commit run --all-files
```

Pytest markers defined in `pytest.ini`: `unit`, `integration`, `end_to_end`, `network`, `glp`, `timeseries`, `validation`, `fast`, `slow`, `memory_intensive`, `cpu_intensive`, `small_data`, `medium_data`, `large_data`, `numerical`, `algorithmic`, `stochastic`, `io`, `external`, `regression`, `smoke`.

## Architecture Essentials

### Three independent feature modules + shared `common/`
- `network/` depends on `common/`
- `glp/` depends on `common/` + `network/`
- `timeseries/` depends on `common/` + `network/`

`glp` and `timeseries` do **not** depend on each other — keep it that way.

The project is currently **backend-only**: no CLI, no REST API, no frontend. Interact with the library through Python imports — notebooks, `test_installation.py`, or ad-hoc scripts. Design new public functions with a future REST API in mind (clear signatures, JSON-serializable returns, no hidden global state), but don't add a CLI layer unless asked.

### IDMapper is load-bearing
NetworkIt requires consecutive integer node IDs (0, 1, 2, …), but inputs use arbitrary identifiers (usernames, UUIDs, …). `common.id_mapper.IDMapper` provides bidirectional mapping, and **every public function uses original IDs externally / internal IDs only inside**. For new functions: accept original IDs, translate via `id_mapper.get_internal_batch()` immediately, translate back before returning.

### GLP algorithm (`glp/propagation.py`)
Matrix-based, sparse, supports both propagation directions. Core update:

```
F⁽ᵗ⁺¹⁾ = α · P · F⁽ᵗ⁾ + (1−α) · Y
```

where `P = D⁻¹A` is the row-normalized transition matrix and `Y` is the seed indicator matrix. Convergence is L∞-norm based. For directed graphs with `directional=True`, propagation runs twice — once with `A` (out-degree / influence) and once with `Aᵀ` (in-degree / receptivity) — returning a `(out_df, in_df)` tuple. Optional features: `enable_noise_category` (auto-adds a "noise" label with random non-seed seeds), `confidence_threshold` (relabels low-confidence nodes as "uncertain").

The predecessor `guidedLP_OLD/network/net_utils.py::stlp` is the reference implementation. The current version replaced its exponential distance decay with degree normalization to fix the class-dominance problem when one class had a few more high-degree seeds (see `conversations.md`).

## Tech Stack Constraints (Hard Rules)

From `docs/tech_requirements.md` and `guidedLP/.clinerules.md`:

1. **NetworkIt**, not NetworkX — all graph ops use the C++ backend.
2. **Polars**, never Pandas — all DataFrame operations.
3. **NumPy + SciPy sparse** for matrix math (sparse for anything graph-sized).
4. **Single-edge graphs only.** No multi-edges, multi-layer, or hypergraphs. Directed/undirected + weighted/unweighted are all supported.
5. **Preserve original IDs end-to-end** (see IDMapper).
6. Speed > memory by default; memory-efficient paths are opt-in.
7. Document Big-O complexity in docstrings for new core functions.
8. Type hints required on public function signatures.

## Import Convention — known inconsistency

Source files under `src/guidedLP/` use absolute imports rooted at the installed package, e.g.

```python
from guidedLP.common.id_mapper import IDMapper
```

Some test files (notably `tests/test_integration.py`) instead use `from src.network.construction import ...`. When editing existing code, match its file's existing style; for new code, prefer the `from guidedLP....` form.

## Reference Documents

For non-trivial changes, consult in order:
1. `guidedLP/docs/architecture/overview.md` — module boundaries, data flow, ID mapping strategy
2. `guidedLP/docs/architecture/glp.md` — GLP algorithm details & directional analysis
3. `guidedLP/docs/specifications/spec_{network,glp,timeseries}.md` — function-level requirements
4. `guidedLP/docs/api/{network,glp,timeseries,common}.md` — public API reference
5. `overall_project_prompt.md` (repo root) — original feature scope
6. `guidedLP/guidedLP_OLD/network/net_utils.py` — reference implementation (`stlp()` in particular)
