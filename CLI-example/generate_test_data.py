import numpy as np

# Oracle: true labels
oracle = np.array([0, 0, 1, 2, 2, 0, 1])
np.save("oracle.npy", oracle)

# Predictions: softmax probabilities (must sum to 1 per row)
predictions = np.array([
    [0.8, 0.1, 0.1],  # correct
    [1.0, 0.0, 0.0],  # correct
    [0.2, 0.7, 0.1],  # correct
    [0.1, 0.2, 0.7],  # correct
    [0.1, 0.4, 0.5],  # correct
    [0.1, 0.8, 0.1],  # wrong
    [0.3, 0.3, 0.4],  # wrong
])
np.save("preds.npy", predictions)

print("Saved oracle.npy and preds.npy")

