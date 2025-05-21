## v1.0.0
Initial release with NetTrustScore.

## v1.0.1
Moved the example image to assets.
Simplified commentary in the scripts.
Updated Readme

## v2.0.0
This release:
1. introduces conditional NetTrustScore (CNTS) that measures trustworthiness for true and false predictions for each class.
2. simplifies the import and call processes, (Trustworthiness.compute_NTS() -> NTS.compute() and CNTS.compute())
3. adds assert conditions,
4. sets trust spectrum plot to False by default,
5. adds plots.py for user interaction under assets,
6. adds conditional_trust_densities.png under assets,
7. has several minor bug fixes and cosmetic improvements

## v2.0.1
Internal changes.

## v2.0.2
1. Published NetTrustScore package to PyPI (pip install nettrustscore now available)
2. Added description to setup.py to fix PyPI/Conda-Forge rendering
3. Added .gitignore to clean build artifacts
4. Synced versioning across Conda-Forge and PyPI
5. No code changes — functionality remains identical to v2.0.0
6. Thanks for using NetTrustScore!

## v2.0.3
1. Package renamed from nettrustscore to trustpy-tools
2. New import path: import trustpy instead of trustquant
3. Ready for Conda-Forge distribution

## v2.0.4
Matching PyPI version. No changes.

## v2.0.5
1. trust_spectrum() now saves Trust spectrum plots and conditional density plots in the appropriate trustpy/nts or trustpy/cnts folders.
2. Plot filenames now auto-append extensions (e.g., .png) if missing.
3. show_summary() now illustrates the results in table form.
4. export_summary() now saves the results in .csv form in the appropriate trustpy/nts or trustpy/cnts folders.
5. Output directories are auto-created for consistent saving.
6. Comprehensive unit tests covering constructor assertions, shape validations, and output correctness for both NTS and CNTS are added via pytest. Tests run via pytest --cov=trustpy with 100% coverage of critical checks.
7. Polished and clarified docstrings and inline comments.
8. Updated README with simplified setup instructions, usage examples, and confirmation that unit tests have already been executed prior to release.

## v2.0.6
Internal changes.

## v2.0.7
1. NTS and CNTS now return float values instead of strings in the .compute() output dictionary.
2. This makes the results easier to use in downstream code, comparisons, and logging — without needing to cast or parse string values.
    Example: {'class_0': 0.667} instead of {'class_0': '0.667'}.
3. Summary printing and CSV exports still round and format values cleanly (.3f) as before.
4. Behavior is fully backward-compatible.

## v2.0.8
This release introduces stricter input validation to improve reliability and prevent silent misuse:
1. Ensures predictions include at least 2 classes.
2. Ensures oracle includes at least 2 unique labels.
3. Enforces match between number of classes in oracle and prediction outputs.
4. Updated error messages for consistency and clarity.
5. Removed single-sample test case (non-generalizable).
6. Adjusted test assertions for compatibility with pytest s match.
7. This is a patch release with no changes to core logic or outputs.

## v2.0.9
1. Included version = "2.0.9" in trustpy/init.py for programmatic version access:
import trustpy
print(trustpy.__version__)  # 2.0.9

2. Added all = ['NTS', 'CNTS'] to restrict from trustpy import * to only public classes.
3. Before writing plots or CSV files, the code now checks for write permissions using:

if not os.access(output_dir, os.W_OK):
    raise PermissionError(...)

This helps prevent runtime crashes on read-only filesystems or restricted environments.

4. In README documentation clarified where plots are saved, added instructions to run tests post-install.
5. Updated incorrect error string in CNTS init from 'Predictions must be a 1D array' → 'Predictions must be a 2D array'.
6. alpha, beta, and other optional parameters in NTS and CNTS constructors must now be passed by keyword.
