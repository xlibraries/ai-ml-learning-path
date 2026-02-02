# visualize.py
import torch
import matplotlib.pyplot as plt
import numpy as np

from model import MLP
from cnn_model import CNN
from dataset import generate_dataset
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_boundary(model, X, y, title, filename):
    # ---- MOVE ONLY FOR PLOTTING ----
    X_cpu = X.cpu()
    y_cpu = y.cpu()

    # Define plotting range (NumPy needs CPU values)
    x_min, x_max = X_cpu[:, 0].min().item() - 0.5, X_cpu[:, 0].max().item() + 0.5
    y_min, y_max = X_cpu[:, 1].min().item() - 0.5, X_cpu[:, 1].max().item() + 0.5

    # Create grid in NumPy (CPU)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Convert grid to tensor and MOVE BACK TO GPU
    grid = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()],
        dtype=torch.float32
    ).to(device)

    # Model inference stays on GPU
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)

    # ---- BACK TO CPU FOR MATPLOTLIB ----
    preds = preds.cpu()

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, preds, levels=50, cmap="coolwarm", alpha=0.6)

    plt.scatter(
        X_cpu[:, 0],
        X_cpu[:, 1],
        c=y_cpu.squeeze(),
        cmap="coolwarm",
        edgecolors="k",
        s=20
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[SAVED] {filename}")

def main():
    # Same dataset for both
    X_train, X_val, y_train, y_val = generate_dataset()
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Train both models
    mlp_model, _, _ = train("mlp")
    cnn_model, _, _ = train("cnn")

    # Visualize decision boundaries
    plot_boundary(
        mlp_model,
        X_val,
        y_val,
        "MLP Decision Boundary",
        "mlp_boundary.png"
    )

    plot_boundary(
        cnn_model,
        X_val,
        y_val,
        "CNN Decision Boundary",
        "cnn_boundary.png"
    )


if __name__ == "__main__":
    main()