# cnn_model.py

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        """
        A minimal Convolutional Neural Network (CNN) for binary classification
        on a 2D input vector (x1, x2).

        Even though CNNs are typically used for images, this model intentionally
        reshapes a 2D vector into a tiny "fake image" to demonstrate how
        convolution can model local feature interactions.

        Input:
            x ∈ ℝ^{B×2}

        Output:
            ŷ ∈ (0,1)^{B×1}

        Key idea:
        ----------
        Instead of treating (x1, x2) independently (as in an MLP),
        we treat them as *adjacent spatial values* and let a convolutional
        kernel learn their interaction.
        """
        super().__init__()

        # -------------------------
        # Convolutional feature extractor
        # -------------------------
        #
        # We reshape the input into shape:
        #   (B, 1, 1, 2)
        #
        # Interpretation:
        # - B: batch size
        # - 1 channel (like grayscale)
        # - height = 1
        # - width = 2  → the two input features sit side-by-side
        #
        # This allows a convolution kernel of size (1×2) to "see"
        # both x1 and x2 at the same time.
        self.conv = nn.Sequential(

            # 2D convolution
            #
            # in_channels = 1:
            #   single input channel
            #
            # out_channels = 8:
            #   learn 8 different feature detectors
            #
            # kernel_size = (1, 2):
            #   spans the full width (x1, x2)
            #
            # Mathematically, each filter learns:
            #   y_k = w_k1 * x1 + w_k2 * x2 + b_k
            #
            # for k = 1..8
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 2)
            ),

            # ReLU activation
            #
            # σ(z) = max(0, z)
            #
            # Adds non-linearity so different filters can activate
            # only for specific relationships between x1 and x2
            nn.ReLU()
        )

        # After convolution:
        #   input  → (B, 1, 1, 2)
        #   output → (B, 8, 1, 1)
        #
        # Each of the 8 channels now represents a learned interaction
        # between x1 and x2.

        # -------------------------
        # Fully connected classifier
        # -------------------------
        #
        # We flatten (B, 8, 1, 1) → (B, 8)
        # and map it to a single probability.
        self.fc = nn.Sequential(

            # Linear layer
            #
            # Takes the 8 learned features and combines them
            # into a single scalar (logit)
            nn.Linear(8, 1),

            # Sigmoid activation
            #
            # σ(z) = 1 / (1 + e^{-z})
            #
            # Converts the logit into a probability
            # suitable for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the CNN.

        Args:
            x (Tensor): Input tensor of shape (B, 2)

        Returns:
            Tensor: Output probabilities of shape (B, 1)
        """

        # -------------------------
        # Reshape input
        # -------------------------
        #
        # Original:
        #   x ∈ ℝ^{B×2}
        #
        # Reshaped to:
        #   x ∈ ℝ^{B×1×1×2}
        #
        # This creates a tiny spatial structure so convolution
        # can be applied.
        x = x.view(-1, 1, 1, 2)

        # -------------------------
        # Convolutional feature extraction
        # -------------------------
        x = self.conv(x)

        # -------------------------
        # Flatten features
        # -------------------------
        #
        # (B, 8, 1, 1) → (B, 8)
        x = x.view(x.size(0), -1)

        # -------------------------
        # Classification head
        # -------------------------
        return self.fc(x)