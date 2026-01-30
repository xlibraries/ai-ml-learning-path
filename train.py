# train.py

# Core PyTorch library
import torch

# Loss functions
import torch.nn as nn

# Optimizers
import torch.optim as optim

# Dataset generator
from dataset import generate_dataset

# Model definition
from model import MLP


def train():
    # Generate training and validation datasets
    X_train, X_val, y_train, y_val = generate_dataset()

    # Instantiate MLP model
    model = MLP()

    # Binary Cross Entropy loss
    #
    # L = -[ y log(ŷ) + (1-y) log(1-ŷ) ]
    #
    # Used for binary classification with sigmoid outputs
    criterion = nn.BCELoss()

    # Adam optimizer
    #
    # Combines:
    #   - Momentum (1st moment)
    #   - RMSProp (2nd moment)
    optimizer = optim.Adam(
        model.parameters(),     # parameters θ to optimize
        lr=0.01,                # learning rate α
        weight_decay=1e-4       # L2 regularization (λ ||θ||²)
    )

    # Lists to track loss curves
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(1000):

        # ---- TRAIN PHASE ----
        model.train()  # enables gradient computation

        # Clear old gradients
        # ∂L/∂θ accumulates by default in PyTorch
        optimizer.zero_grad()

        # Forward pass
        # ŷ = f(x; θ)
        y_pred = model(X_train)

        # Compute training loss
        loss = criterion(y_pred, y_train)

        # Backpropagation
        #
        # Computes:
        #   ∂L/∂θ using chain rule
        loss.backward()

        # Parameter update
        #
        # θ ← θ - α ∇θ L
        optimizer.step()

        # ---- VALIDATION PHASE ----
        model.eval()  # disables dropout / batchnorm (if any)

        # Disable gradient tracking for validation
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        # Store losses for visualization
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss.item():.4f}"
            )

    # Save trained weights to disk
    torch.save(model.state_dict(), "mlp_model.pth")

    return train_losses, val_losses


# Entry point
if __name__ == "__main__":
    train()# model.py