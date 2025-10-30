import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumClassifier:
    """
    Classical feed‑forward classifier mirroring the interface of the quantum helper.

    Parameters
    ----------
    num_features : int
        Number of input features / qubits.
    depth : int
        Number of hidden layers.
    dropout : float, optional
        Dropout probability applied after each hidden layer (default 0.0, i.e. no dropout).
    use_batchnorm : bool, optional
        Whether to insert a BatchNorm1d after each hidden layer.
    """

    def __init__(self, num_features: int, depth: int, dropout: float = 0.0, use_batchnorm: bool = False):
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.model, self.encoding, self.weight_sizes, self.observables = self._build_model()

    def _build_model(self):
        layers = []
        in_dim = self.num_features
        encoding = list(range(self.num_features))
        weight_sizes = []

        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            layers.append(nn.ReLU())
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.num_features))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = self.num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        model = nn.Sequential(*layers)
        observables = list(range(2))  # class indices
        return model, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

__all__ = ["QuantumClassifier"]
