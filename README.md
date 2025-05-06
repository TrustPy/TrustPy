# TrustPy - Trustworthiness Python

Python package for validating AI/ML model reliability and uncertainty during development and before deployment. Distributed via Conda-Forge and PyPI.

###
**The implementation is flexible and works out-the-box with any AI/ML library.**
###

## Installation
### Recommended 1: Install via Conda-Forge
The easiest way to install trustpy-tools is via Conda-Forge, which handles all dependencies automatically. Run the following command:
```bash
conda install -c conda-forge trustpy-tools
```

### Recommended 2: Install via PyPI (pip install)
If you prefer using pip (PyPI), you can install directly:
```bash
pip install trustpy-tools
```

### Alternative: Manual Installation
If you prefer to install the package manually or are not using Conda, you can install the required dependencies and clone the repository.

Install Dependencies
- **NumPy**: For numerical calculations.
- **Matplotlib**: For plotting the trust spectrum.
- **Scikit-learn**: For Kernel Density Estimation (KDE) in trust density estimation.

Install them via conda:

```bash
conda install numpy matplotlib scikit-learn
```

or

Install them via pip:

```bash
pip install numpy matplotlib scikit-learn
```

Clone the Repository
```bash
git clone https://github.com/yaniker/TrustPy.git
cd TrustPy
```

You can verify installation by running:
```bash
python -c "from trustpy import NTS, CNTS; print('TrustPy is ready.')"
```

## Example Usage
```python
from trustpy import NTS, CNTS #This is how the package is imported.
import numpy as np

# Example oracle and predictions
oracle = np.array([0, 0, 1, 2, 2, 0, 1])  # True labels
predictions = np.array([
    [0.8, 0.1, 0.1],  # Correct, high confidence
    [1.0, 0.0, 0.0],  # Correct, high confidence
    [0.2, 0.7, 0.1],  # Correct, high confidence
    [0.1, 0.2, 0.7],  # Correct, high confidence
    [0.1, 0.4, 0.5],  # Correct, lower confidence
    [0.1, 0.8, 0.1],  # Incorrect, high confidence
    [0.3, 0.3, 0.4]   # Incorrect, low confidence
]
) #Replace this with your model's predictions (`predictions = model.predict()`)

# FOR NETTRUSTSCORE #
# Initialize with default parameters
nts = NTS(oracle, predictions, show_summary=True, export_summary=True, trust_spectrum=True)
nts_scores_dict = nts.compute() # Computes trustworthiness for each class and overall.

# FOR CONDITIONAL NETTRUSTSCORE #
# Initialize with default parameters
cnts = CNTS(oracle, predictions, , show_summary=True, export_summary=True, trust_spectrum=True)
cnts_scores_dict = cnts.compute() # Computes trustworthiness for each class and overall.

# Sets show_summary=True to print the results table.
# Sets export_summary=True to save the results.
# Sets trust_spectrum=True to generate trust_spectrum.png
```

Example Plot for Trust Spectrum (`trust_spectrum = True`)
![Alt text](./assets/trust_spectrum.png)

Example Plot for Conditional Trust Spectrum (`trust_spectrum = True`)
![Alt text](./assets/conditional_trust_densities.png)

I shared the codes for the plots [Python scripts for plots](./assets/plots.py) for users to modify as needed.

## Unit Testing
All unit tests were run using `pytest` with full coverage prior to release to ensure reliability and correctness.

## Licence
This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## Citations
For scholarly references and the origins of the techniques used in this package, please refer to the [CITATION](./CITATION.cff) file.
