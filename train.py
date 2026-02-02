# train.py
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import generate_dataset
from model import MLP
from cnn_model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model_type="mlp"):
    X_train, X_val, y_train, y_val = generate_dataset()

    # Move data to GPU / CPU
    X_train, X_val = X_train.to(device), X_val.to(device)
    y_train, y_val = y_train.to(device), y_val.to(device)

    # Select model
    if model_type == "mlp":
        model = MLP().to(device)
        save_path = "mlp_model.pth"
    elif model_type == "cnn":
        model = CNN().to(device)
        save_path = "cnn_model.pth"
    else:
        raise ValueError("Unknown model type")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.05,
        weight_decay=1e-4
    )

    train_losses, val_losses = [], []

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 100 == 0:
            print(
                f"[{model_type.upper()}] "
                f"Epoch {epoch:4d} | "
                f"Train {loss.item():.4f} | "
                f"Val {val_loss.item():.4f}"
            )

    torch.save(model.state_dict(), save_path)
    return model, train_losses, val_losses


if __name__ == "__main__":
    train("mlp")
    train("cnn")