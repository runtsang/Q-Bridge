import torch
import torch.nn as nn
import numpy as np

class HybridQuantumFullyConnectedClassifier(nn.Module):
    """
    Classical approximation of a hybrid fully‑connected quantum layer.
    Replaces the quantum expectation head with a learnable tanh‑activated
    linear layer. Provides a run method that mimics the interface of the
    quantum implementation.
    """
    def __init__(self, n_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Mimic the quantum circuit run interface.
        `thetas` is expected to be a 1‑D array of shape (n_features,).
        The method returns a probability array of shape (2,).
        """
        with torch.no_grad():
            tensor = torch.as_tensor(thetas, dtype=torch.float32).unsqueeze(0)
            logits = self.linear(tensor)
            probs = torch.sigmoid(logits + self.shift)
            return probs.squeeze().numpy()
