import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], int]:
    """
    Construct a feed‑forward classifier/regressor that mirrors the quantum
    circuit interface.

    Returns
    -------
    network : nn.Sequential
        Classical trunk + head.
    encoding : List[int]
        Encoding indices (identical to the quantum encoder).
    weight_sizes : List[int]
        Number of parameters per linear layer, used for analysis or
        hybrid weight transfer.
    output_dim : int
        Size of the final prediction vector (1 for scalar regression,
        2 for binary classification).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    # Build trunk
    for _ in range(depth):
        lin = nn.Linear(in_dim, num_features)
        layers.append(lin)
        layers.append(nn.ReLU())
        weight_sizes.append(lin.weight.numel() + lin.bias.numel())
        in_dim = num_features

    # Head
    head = nn.Linear(in_dim, 1)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)

    # Default to regression; binary classification can be achieved by
    # adding a sigmoid or softmax externally.
    output_dim = 1
    return network, encoding, weight_sizes, output_dim

__all__ = ["build_classifier_circuit"]

class UnifiedClassifierRegressor(nn.Module):
    """
    Hybrid‑inspired classical trunk that can optionally expose quantum
    parameters for a downstream Qiskit circuit.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    depth : int, default=2
        Number of hidden layers in the trunk.
    use_quantum : bool, default=False
        If True, the module will expose a ParameterVector that can be
        bound to a Qiskit circuit.  The forward method remains purely
        classical; the quantum component is expected to be executed
        externally.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 use_quantum: bool = False):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_quantum = use_quantum

        # Classical trunk
        self.trunk = nn.Sequential(
            *[nn.Sequential(
                 nn.Linear(num_features, num_features),
                 nn.ReLU()
             ) for _ in range(depth)]
        )

        # Head
        self.head = nn.Linear(num_features, 1)

        # Quantum parameters placeholder
        if self.use_quantum:
            from qiskit.circuit import ParameterVector
            self.quantum_params = ParameterVector("theta", num_features * depth)
        else:
            self.quantum_params = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  If ``use_quantum`` is True, the method still
        returns the classical output; the quantum layer is expected to
        be applied in a separate training loop.
        """
        out = self.trunk(x)
        out = self.head(out)
        return out.squeeze(-1)

    @staticmethod
    def build_encoder(num_features: int) -> Tuple[List[int], List[int]]:
        """
        Return the encoding indices and weight size metadata that
        match the quantum circuit's expectation.
        """
        encoding = list(range(num_features))
        weight_sizes = [num_features * num_features] * 2
        return encoding, weight_sizes

    @property
    def weight_metadata(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Expose weight sizes for trunk and head layers.
        """
        trunk_w = [layer.weight.numel() + layer.bias.numel()
                   for layer in self.trunk]
        head_w = [self.head.weight.numel() + self.head.bias.numel()]
        return trunk_w, head_w, []

    def train(self, mode: bool = True, *, non_blocking: bool = False):
        """
        Keep compatibility with the reference API.
        """
        super().train(mode, non_blocking=non_blocking)
