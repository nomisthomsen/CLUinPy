# Changelog

## V2.0 (Rename to CLUinPy)

### Changed
- Project renamed from **CLUMondoPy** to **CLUinPy** to avoid confusion with the original CLUMondo model.
- Python package renamed from `CLUMondo` to `CLUinPy`.
- Updated imports, scripts, and documentation to reflect the new naming.

### Renamed
- Renamed repository/project folder: `CLUMondoPy` → `CLUinPy`.
- Renamed main run script: `run_CLUMondoPy.py` → `run_CLUinPy.py`.
- Updated filenames and references in documentation/manuals where applicable.

### Fixed
- Removed generated Python bytecode caches (`__pycache__/`, `*.pyc`) from version control and added them to `.gitignore`.
- Removed PyCharm project metadata (`.idea/`, `*.iml`) from version control and added them to `.gitignore`.

### Breaking changes
- Import paths changed:
  - **Before:** `import CLUMondo` / `from CLUMondo ...`
  - **After:**  `import CLUinPy` / `from CLUinPy ...`
- Any external scripts or notebooks referencing the old names must be updated.
