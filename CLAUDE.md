# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XManager is DeepMind's framework for packaging, running, and tracking machine learning experiments. It supports local execution, Google Cloud (Vertex AI), and Kubernetes backends. Launch scripts are Python programs that configure experiments using an eager execution model (no YAML/DSL).

## Common Commands

### Install
```bash
pip install -e .
```

### Run tests
Tests use `absl.testing` (absltest) and `unittest`. Test files follow the `*_test.py` pattern and are colocated with source modules.

```bash
# Run a single test file
python -m xmanager.xm.core_test
python -m xmanager.xm.resources_test

# Run all tests (no unified test runner configured; run files individually)
python -m pytest xmanager/ -k '_test'
```

### Launch an experiment
```bash
xmanager launch examples/<example_script>.py -- [--args]
```

### Compile Protocol Buffers
```bash
bash compile_pb.sh
```

## Architecture

### Package Structure

- **`xmanager.xm`** — Core public API. Abstract base classes for `Experiment`, `WorkUnit`, `Executor`, `ExecutableSpec`. Users import `from xmanager import xm`.
- **`xmanager.xm_local`** — Local backend implementation. Users import `from xmanager import xm_local` for `create_experiment()`, executor types (`Local`, `Vertex`, `Kubernetes`).
- **`xmanager.cloud`** — GCP integration: Vertex AI, Cloud Build, Docker image building, Kubernetes, authentication.
- **`xmanager.cli`** — CLI entry point (`xmanager` command). Entry point: `xmanager.cli.cli:entrypoint`.
- **`xmanager.contrib`** — Optional utilities: GCS helpers, TensorBoard, TensorFlow defaults, parameter controllers.
- **`xmanager.xm_mock`** — Mock implementations for testing launch scripts without real backends.
- **`xmanager.generated`** — Auto-generated Protocol Buffer files. Do not edit manually.
- **`xmanager.module_lazy_loader`** — Lazy import system to reduce CLI startup time.

### Key Design Patterns

**Abstract API / Backend split**: `xm.core` defines abstract `Experiment`, `WorkUnit`, `ExperimentUnit`. `xm_local` provides concrete implementations. New backends override these abstract methods.

**Lazy loading**: `xm_local/__init__.py` uses `XManagerLazyLoader` to defer imports until attribute access. When adding new public APIs to `xm_local`, add an `XManagerAPI` entry to `_apis` list AND a corresponding `TYPE_CHECKING` import block.

**Packaging router**: `xm_local/packaging/router.py` dispatches packaging to the correct handler (local vs cloud) based on executor type.

**Executor registry**: `xm_local/registry.py` dynamically registers executors. Packaging and execution logic dispatches based on registered executor types.

**Context variables**: `contextvars` in `xm/core.py` track the current experiment and work unit, enabling implicit context in job generators.

### Database & Persistence

SQLAlchemy ORM (`xm_local/storage/database.py`) with Alembic migrations (`xm_local/storage/alembic/`). Protocol Buffers (`xm_local/storage/data.proto`) for serialization.

## Key Conventions

- Python >= 3.10 required (uses `match` statements)
- Uses `attr` (attrs library) for data classes throughout
- Google-style imports: `from xmanager.xm import core` rather than relative imports
- Test files use `absltest.main()` or `unittest.main()` as entry points
