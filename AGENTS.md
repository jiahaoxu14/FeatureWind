# Repository Guidelines

## Project Structure & Module Organization
- `src/featurewind/`: Core library (tangent maps, gradient/DR utilities, data types). Import via `PYTHONPATH=src` or add `src` to `sys.path` in scripts.
- `examples/`: Runnable visualization and utilities. Recommended entry point: `examples/main_modular.py`; data prep: `examples/generate_tangent_map.py`.
- `tangentmaps/`: Sample `.tmap` data files used by examples.
- `output/`: Generated frames/animations/CSVs (git‑ignored; safe for large outputs).
- `archive/`: Older/legacy prototypes kept for reference.

## Build, Test, and Development Commands
- Run modular viz: `python examples/main_modular.py`
- Run legacy viz: `python examples/main.py`
- Generate `.tmap` from CSV: `python examples/generate_tangent_map.py data.csv tsne --target <label> --output tangentmaps/data.tmap`
- Direct (legacy) generator: `python src/featurewind/TangentMap.py <input.csv> tsne`
Tip: Use a virtualenv (e.g., `.venv`) and Python 3.9+.

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indentation.
- Modules/files: `snake_case.py`; functions/vars: `snake_case`; classes: `PascalCase`.
- Keep modules focused (mirroring `examples/*` modularization). Prefer explicit over implicit imports.
- Type hints and short docstrings for new/edited public functions.

## Testing Guidelines
- No formal test suite yet. Validate changes by:
  - Running `examples/main_modular.py` against a known `.tmap` in `tangentmaps/`.
  - Exercising `examples/generate_tangent_map.py` on a small CSV and loading the result.
- If adding tests, use `pytest`, place under `tests/`, name `test_*.py`.

## Commit & Pull Request Guidelines
- Commits: short, imperative, scoped (e.g., "Add single feature mode", "Optimize particle system"). Group related edits.
- PRs: include a concise description, linked issues, repro steps, before/after screenshots or short clips when UI/visual changes are involved.
- Keep PRs focused; note follow‑ups explicitly.

## Security & Configuration Tips
- Do not commit large artifacts; `output/` is git‑ignored. Prefer adding sample `.tmap` to `tangentmaps/` when needed.
- Paths in examples are relative; keep data under `tangentmaps/` or pass explicit paths.
- Dependencies commonly used: PyTorch, NumPy/SciPy, Matplotlib, scikit‑learn. Pin versions in your environment as needed.

