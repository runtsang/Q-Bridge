import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

# Classical factory mirroring the quantum API
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with metadata compatible with the quantum variant.
    Returns (network, encoding, weight_sizes, observables).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = [0, 1]  # placeholder for two‑class logits
    return network, encoding, weight_sizes, observables


class QuantumHybridClassifier(nn.Module):
    """
    Hybrid classifier that couples a classical backbone with an optional quantum feature extractor.
    The API matches the original QuantumClassifierModel for seamless integration.
    """

    def __init__(self, num_features: int, depth: int = 4, quantum_enabled: bool = True):
        super().__init__()
        self.classical_net, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )
        self.quantum_enabled = quantum_enabled
        self.quantum_module: nn.Module | None = None
        self.quantum_head: nn.Module | None = None

        if quantum_enabled:
            # Placeholder: the quantum module must be set externally via `set_quantum_module`.
            self.quantum_head = nn.Linear(num_features, 2)

    def set_quantum_module(self, qm: nn.Module):
        """
        Attach a quantum module that outputs a feature vector.
        The module must expose an `output_dim` attribute.
        """
        self.quantum_module = qm
        in_features = self.classical_net[-1].out_features + qm.output_dim
        self.quantum_head = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantum fusion.
        """
        if self.quantum_enabled and self.quantum_module is not None:
            # Classical pathway
            cl_features = self.classical_net(x)
            # Quantum pathway
            q_features = self.quantum_module(x)
            # Simple concatenation fusion
            fused = torch.cat([cl_features, q_features], dim=1)
            logits = self.quantum_head(fused)
        else:
            logits = self.classical_net(x)
        return logits


__all__ = ["QuantumHybridClassifier", "build_classifier_circuit"]
