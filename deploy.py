# deploy.py

import torch
from model import MLP


class MLPService:
    def __init__(self, model_path="mlp_model.pth"):
        # Initialize model
        self.model = MLP()

        # Load trained parameters
        self.model.load_state_dict(torch.load(model_path))

        # Switch to inference mode
        self.model.eval()

    def predict(self, x):
        # Convert input to tensor
        x = torch.tensor(x, dtype=torch.float32)

        # Disable gradient tracking
        with torch.no_grad():
            # Forward pass and return NumPy output
            return self.model(x).numpy()


# Example usage
if __name__ == "__main__":
    service = MLPService()
    print(service.predict([[0.2, -0.1]]))