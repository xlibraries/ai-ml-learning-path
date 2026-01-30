# dataset.py

# PyTorch tensor library
# Used to store data as tensors and later move to GPU if needed
import torch

# make_moons generates a classic 2D non-linear classification dataset
# This dataset is intentionally NOT linearly separable
from sklearn.datasets import make_moons

# Utility to split data into training and validation sets
from sklearn.model_selection import train_test_split


def generate_dataset(n_samples=1000, noise=0.2, test_size=0.2):
    # Generate a synthetic dataset of two interleaving half-circles ("moons")
    # X ∈ ℝ^{n×2}, y ∈ {0,1}
    #
    # Mathematically:
    #   Each sample x_i = (x1, x2) lies in 2D
    #   Labels depend on a non-linear boundary
    X, y = make_moons(n_samples=n_samples, noise=noise)

    # Convert NumPy array → PyTorch tensor
    # dtype=float32 because neural networks operate in floating point
    X = torch.tensor(X, dtype=torch.float32)

    # Convert labels to float tensor
    # unsqueeze(1) turns shape from (N,) → (N,1)
    #
    # This is REQUIRED because:
    #   - BCELoss expects predictions and targets of the same shape
    #   - Our model outputs (N,1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Split dataset into training and validation sets
    #
    # random_state ensures reproducibility:
    # same split every run → stable debugging
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Return tensors ready for training
    return X_train, X_val, y_train, y_val