# inference.py

# PyTorch
import torch

# Model architecture
from model import MLP


def load_model():
    # Create model instance
    model = MLP()

    # Load trained parameters θ*
    model.load_state_dict(torch.load("mlp_model.pth"))

    # Set model to inference mode
    model.eval()

    return model


def predict(x):
    # Load trained model
    model = load_model()

    # Convert input to tensor
    # Shape: (N,2)
    x = torch.tensor(x, dtype=torch.float32)

    # Disable gradient computation
    with torch.no_grad():
        # Forward pass
        # Output is probability
        prob = model(x)

    return prob


if __name__ == "__main__":
    # Example inference point
    sample = [[0.5, 0.2]]

    # Print predicted probability
    print("Prediction:", predict(sample).item())