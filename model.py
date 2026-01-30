# model.py

# PyTorch neural network module base classes
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        """
        Initialize a 3-layer neural network for binary classification.

        Args:
            input_dim (int, optional): Dimension of input features. Defaults to 2.
            hidden_dim (int, optional): Dimension of hidden layers. Defaults to 32.

        The parameter `hidden_dim=32` specifies the number of neurons in the two hidden layers.
        It's an arbitrary choice that balances:
        - Model capacity: More neurons (larger 32) = more expressive, can learn complex patterns
        - Computational cost: More neurons = slower training and more memory
        - Overfitting risk: More neurons = higher risk of overfitting on small datasets

        The value 32 is a common "sweet spot" for small problems:
        - Not too small (like 8) to lose expressiveness
        - Not too large (like 512) to waste computation
        - Works well for 2D input classification tasks

        You can experiment with different values (16, 64, 128, etc.) to find what works best
        for your specific dataset and problem.
        """
        # Initialize nn.Module internals
        super().__init__()

        # Define the network as a Sequential stack
        #
        # This is mathematically:
        #   f(x) = σ(W3 · σ(W2 · σ(W1 · x)))
        #
        # Where:
        #   W1 ∈ ℝ^{32×2}
        #   W2 ∈ ℝ^{32×32}
        #   W3 ∈ ℝ^{1×32}
        self.net = nn.Sequential(

            # First affine transformation
            #
            # y = W1 x + b1
            # Projects 2D input into a 32D feature space
            nn.Linear(input_dim, hidden_dim),

            # ReLU non-linearity
            #
            # σ(z) = max(0, z)
            # Introduces non-linearity so the model can learn curved boundaries
            nn.ReLU(),

            # Second affine transformation
            #
            # Allows the network to combine intermediate features
            nn.Linear(hidden_dim, hidden_dim),

            # Another non-linearity
            #
            # Without this, stacked linear layers would collapse into one
            nn.ReLU(),

            # Final affine layer
            #
            # Maps features → single scalar (logit)
            nn.Linear(hidden_dim, 1),

            # Sigmoid activation
            #
            # σ(z) = 1 / (1 + e^{-z})
            # Converts logit into probability ∈ (0,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        # Forward pass through entire network
        #
        # Input:  x ∈ ℝ^{N×2}
        # Output: ŷ ∈ (0,1)^{N×1}
        return self.net(x)