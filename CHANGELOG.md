All notable changes to this project are documented here.

## Version 2.0.10 – Maintenance Release
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
