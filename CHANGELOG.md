All notable changes to this project are documented here.
## Version 2.0.14
Release Date: 2025-06-21

Introduced --output_dir flag to the CLI, allowing users to specify a custom directory for saving plots and CSV summaries.
Both NTS and CNTS now accept an output_dir argument to programmatically control output location.
If output_dir is not set, results default to:

trustpy/nts/ (for NTS)
trustpy/cnts/ (for CNTS)

Added unit tests and CLI tests to ensure output is correctly saved and isolated.
Updated README examples, CLI section, and usage notes.

## Version 2.0.13
Release Date: 2025-06-07  

Updated CLI help message for `--mode`. Now it clearly explains selecting nts and cnts with reasoning.
Added `.bumpversion.cfg` for consistent version management across files.
Integrated `flake8` linting into CI for code quality enforcement.

!!! **Upcoming in v3.0.0** !!!  

The next release will be a major update that restructures the codebase to adopt full scikit-learn API conventions to ensure compatibility with scikit-learn pipelines and common conventions. 
TrustPy v2.x will remain usable, but will no longer receive new features once v3.0.0 is released.


## Version 2.0.12
Release Date: 2025-06-02  

Added `__repr__()` methods to `NTS` and `CNTS` classes for object introspection.
Objects now display key configuration details when printed.

## Version 2.0.11.post1 (build: 1 for Conda-forge)
Release Date: 2025-05-30  

Updated project URL and metadata to reflect transfer to new GitHub organization: [`TrustPy/TrustPy`](https://github.com/TrustPy/TrustPy)
No changes to code, logic, or dependencies.
Repackaged and re-released on PyPI under PEP 440-compliant post-release tag `2.0.11.post1`.

## Version 2.0.11

Release Date: 2025-05-29  

Added CLI support via `python -m trustpy` and installed entry point (trustpy) for direct command-line execution.
CLI accepts --oracle, --pred, --mode, --trust_spectrum, and --no_summary flags.
Introduced CLI unit tests in tests/test_cli.py.
Plots generated via CLI (--trust_spectrum) are saved to:
`trustpy/nts/trust_spectrum.png` (for NTS)
`trustpy/cnts/trust_spectrum.png` and `conditional_trust_densities.png` (for CNTS)
Updated `trustpy/__init__.py` to reflect version 2.0.11.
Updated `README` with CLI usage instructions.

## Version 2.0.10
Release Date: 2025-05-21

Migrated from os.path to pathlib.Path for all file and plot path handling (in NTS, CNTS, and plot scripts).
Added post-installation one-liner test commands in the README to verify package functionality and plot output. These validate both in-memory logic and disk write permissions using real synthetic data.
Introduced MANIFEST.in to ensure README.md, LICENSE, and CITATION.cff are bundled with PyPI/Conda-Forge distributions.

## v2.0.9
Release Date: 2025-05-19

Included __version__ = "2.0.9" in trustpy/__init__.py for programmatic version access.
Added __all__ = ['NTS', 'CNTS'] to restrict from trustpy import *.
Added os.access(..., os.W_OK) checks before file writes to avoid runtime errors.
Clarified plot output paths and added pytest usage to README.
Fixed misleading error messages (e.g., “Predictions must be a 1D array” → “2D array”).
Made alpha, beta, and other optional args keyword-only in constructors.

## v2.0.8
Release Date: 2025-05-13

Input validation enhancements:
Requires at least 2 classes in oracle and prediction arrays.
Enforces oracle/prediction class count match.
Clear, consistent assertion error messages.
Removed unsupported single-sample test.
Compatible with pytest's match argument.

## v2.0.7
Release Date: 2025-05-09

compute() now returns raw float values instead of strings.
CSV output and printed summaries retain rounded formatting.
Fully backward-compatible.

## v2.0.6
Release Date: 2025-05-09

Internal changes only.

## v2.0.5
Release Date: 2025-05-06

Auto-creates output folders (trustpy/nts, trustpy/cnts).
Saves plots and CSVs with safe extensions if not provided.
Enhanced test coverage with pytest --cov.
Improved documentation and setup instructions.

## v2.0.4
Version bump to match PyPI release.

## v2.0.3
Package renamed to trustpy-tools.
Changed import path: import trustpy.

## v2.0.2
Published to PyPI.
Fixed metadata display on PyPI/Conda.
Synced versioning.
No logic changes.

## v2.0.1
Release Date: 2025-04-01

Internal changes.

## v2.0.0
Release Date: 2025-04-01

Major update:
Introduced Conditional NetTrustScore (CNTS).
Unified call signatures: .compute() for both NTS and CNTS.
Added input assertions, default trust spectrum False.
Added plots.py and visual output under assets/.
Multiple visual and usability improvements.

##v1.0.1
Release Date: 2025-03-30

Moved image to assets/.
Cleaned script comments.
Updated README.

## v1.0.0
Release Date: 2025-03-29

Initial release with NetTrustScore (NTS).
