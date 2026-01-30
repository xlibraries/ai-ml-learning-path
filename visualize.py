# visualize.py

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import MLP
from dataset import generate_dataset
from train import train


def plot_decision_boundary(model, X, y, title, filename):
    """
    Plots and SAVES the decision boundary.

    model    : trained or untrained MLP
    X, y     : dataset
    title    : plot title
    filename: png file to save
    """

    # Define plotting range with padding
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create dense grid over input space
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Convert grid to tensor: (N, 2)
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    )

    # Forward pass on grid
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)

    # Plot decision surface
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds, levels=50, cmap="coolwarm", alpha=0.6)

    # Overlay dataset points
    plt.scatter(
        X[:, 0], X[:, 1],
        c=y.squeeze(),
        cmap="coolwarm",
        edgecolors="k",
        s=20
    )

    plt.title(title)
    plt.tight_layout()

    # SAVE instead of show (headless-safe)
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[SAVED] {filename}")


def main():
    # Generate dataset ONCE (important for fair comparison)
    X_train, X_val, y_train, y_val = generate_dataset()

    # --------------------------------------------------
    # 1️⃣ BEFORE TRAINING (random weights)
    # --------------------------------------------------
    untrained_model = MLP()

    plot_decision_boundary(
        model=untrained_model,
        X=X_train,
        y=y_train,
        title="Decision Boundary (Before Training)",
        filename="decision_boundary_before_training.png"
    )

    # --------------------------------------------------
    # 2️⃣ TRAIN MODEL
    # --------------------------------------------------
    train_losses, val_losses = train()

    # Reload trained model
    trained_model = MLP()
    trained_model.load_state_dict(torch.load("mlp_model.pth"))

    # --------------------------------------------------
    # 3️⃣ LOSS CURVES
    # --------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig("loss_curves.png", dpi=150)
    plt.close()

    print("[SAVED] loss_curves.png")

    # --------------------------------------------------
    # 4️⃣ AFTER TRAINING (learned boundary)
    # --------------------------------------------------
    plot_decision_boundary(
        model=trained_model,
        X=X_val,
        y=y_val,
        title="Decision Boundary (After Training)",
        filename="decision_boundary_after_training.png"
    )


if __name__ == "__main__":
    main()