# How Much Can We Really Trust You? Towards Simple, Interpretable Trust Quantification Metrics for Deep Neural Networks

This project provides a Python implementation of trustworthiness quantification metrics for predictive models (e.g., DNNs) as proposed in [How Much Can We Really Trust You? Towards Simple, Interpretable Trust Quantification Metrics for Deep Neural Networks](Wong, A., Wang, X. Y., &amp; Hryniowski, A. (2020). How much can we really trust you? towards simple, interpretable trust quantification metrics for deep neural networks. arXiv preprint arXiv:2009.05835). The metrics evaluate a model's trustworthiness based on its confidence behavior in correct and incorrect predictions.  

###
**The implementation is flexible and can be used with any Python code that produces softmax probabilities for classification tasks without modifications.**
###

### Key Features
- **Question-Answer Trust**: Computes trust scores per sample based on prediction correctness and softmax probability.
- **Trust Density**: Estimates the distribution of trust scores per class using Kernel Density Estimation (KDE).
- **Trust Spectrum**: Visualizes trust densities across all classes.
- **NetTrustScore (NTS)**: Provides a scalar metric summarizing overall trustworthiness.

These tools are ideal for assessing and comparing the **reliability** of predictive models (e.g., DNNs) in classification scenarios.

---

## Installation

To use this code, install the required Python packages:

- **NumPy**: For numerical calculations.
- **Matplotlib**: For plotting the trust spectrum.
- **Scikit-learn**: For Kernel Density Estimation (KDE) in trust density estimation.

Install them via pip:

```bash
pip install numpy matplotlib scikit-learn
```

or

Install them via conda:

```bash
conda install numpy matplotlib scikit-learn
```

## Example
```python
import numpy as np
from trustworthiness import Trustworthiness

# Example oracle and predictions
oracle = np.array([0, 1, 2, 0, 1])  # True labels
predictions = np.array([
    [0.8, 0.1, 0.1],  # Correct, high confidence
    [0.2, 0.7, 0.1],  # Correct, high confidence
    [0.1, 0.2, 0.7],  # Correct, high confidence
    [0.1, 0.8, 0.1],  # Incorrect, high confidence for wrong class
    [0.3, 0.3, 0.4]   # Incorrect, low confidence
]
) #Replace this with your model's predictions (`predictions = model.predict()`)

# Initialize with default parameters
trust = Trustworthiness(oracle, predictions)

# Compute trustworthiness metrics. trust_spectrum = True if the user wants spectrum plot to be saved.
class_nts, overall_nts = trust.compute_NTS(trust_spectrum = True)
print("NTS, per class:", class_nts)
print("NTS, overall  :", overall_nts)

```

## I developed this code in 2021 as a part of my publications as listed below:  
1. Yanik, E., Kruger, U., Intes, X., Rahul, R., & De, S. (2023). Video-based formative and summative assessment of surgical tasks using deep learning. Scientific Reports, 13(1), 1038.
2. Yanik, E., Ainam, J. P., Fu, Y., Schwaitzberg, S., Cavuoto, L., & De, S. (2024). Video-based skill acquisition assessment in laparoscopic surgery using deep learning. Global Surgical Education-Journal of the Association for Surgical Education, 3(1), 26.
3. Yanik, E., Schwaitzberg, S., Yang, G., Intes, X., Norfleet, J., Hackett, M., & De, S. (2024). One-shot skill assessment in high-stakes domains with limited data via meta learning. Computers in Biology and Medicine, 174, 108470. / Yanik, E., Schwaitzberg, S., Yang, J., Intes, X., & De, S. (2022). One-shot domain adaptation in video-based assessment of surgical skills. arXiv e-prints, arXiv-2301.
4. Yanik, E., Intes, X., & De, S. (2024). Cognitive-Motor Integration in Assessing Bimanual Motor Skills. arXiv preprint arXiv:2404.10889.
5. Ainam, J. P., Yanik, E., Rahul, R., Kunkes, T., Cavuoto, L., Clemency, B., ... & De, S. (2024). Deep Learning for Video-Based Assessment of Endotracheal Intubation Skills. arXiv preprint arXiv:2404.11727.
