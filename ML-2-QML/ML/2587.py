import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionRegressionModel(nn.Module):
    """
    Classical hybrid model that applies a 2x2 convolutional filter to 28x28 images
    and produces both classification logits and a regression output.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 2x2 conv reduces spatial size from 28x28 to 14x14, 1 input channel to 4 feature maps
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Flattened feature dimension: 4 * 14 * 14
        self.feature_dim = 4 * 14 * 14
        # Classification head
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        # Regression head
        self.regressor = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        dict
            Dictionary with keys 'logits' (log-softmax over classes)
            and'regression' (scalar regression output).
        """
        features = self.conv(x)                     # (batch, 4, 14, 14)
        flat = features.view(features.size(0), -1)  # (batch, feature_dim)
        logits = self.classifier(flat)
        regression = self.regressor(flat).squeeze(-1)
        return {
            "logits": F.log_softmax(logits, dim=-1),
            "regression": regression,
        }

__all__ = ["HybridQuanvolutionRegressionModel"]
