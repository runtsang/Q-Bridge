"""Classical hybrid classifier with optional quantum transformer.

The module defines a `HybridQuantumClassifier` that mirrors the API of the
original `build_classifier_circuit` function but adds a flag to optionally
insert a quantum transformer block.  The classical implementation remains
fully PyTorch compatible, while the quantum variant can be swapped in by
importing from the `qml` module.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

class HybridQuantumClassifier(nn.Module):
    """
    Classical feed‑forward classifier with optional quantum transformer.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden linear layers.
    use_qtransform : bool, default False
        If True, a quantum transformer block is inserted after the hidden
        layers.  The block is lazily imported from the quantum module.
    """

    def __init__(self, num_features: int, depth: int, use_qtransform: bool = False):
        super().__init__()
        self.use_qtransform = use_qtransform

        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features

        self.hidden = nn.Sequential(*layers)

        if use_qtransform:
            # Lazy import to keep the module purely classical until needed
            from.quantum_transformer import QuantumTransformerBlock  # type: ignore

            self.qtransform = QuantumTransformerBlock(
                embed_dim=num_features,
                num_heads=4,
                ffn_dim=2 * num_features,
                n_qubits=8,
                device="cpu",
            )
        else:
            self.qtransform = None

        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        if self.qtransform is not None:
            # Flatten the features into a sequence for the transformer
            seq_len = 1
            x = x.unsqueeze(1).repeat(1, seq_len, -1)
            x = self.qtransform(x)
            x = x.mean(dim=1)  # pool over sequence
        return self.head(x)

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_classifier_circuit(
    num_features: int,
    depth: int,
    use_qtransform: bool = False,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a hybrid classifier and return metadata.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden layers in the classical part.
    use_qtransform : bool
        Whether to include the quantum transformer block.

    Returns
    -------
    nn.Module
        The classifier instance.
    Iterable[int]
        Encoding indices (identity mapping).
    Iterable[int]
        Per‑layer parameter counts.
    List[int]
        Observable indices (class labels).
    """
    classifier = HybridQuantumClassifier(num_features, depth, use_qtransform)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in classifier.parameters() if p.requires_grad]
    observables = [0, 1]
    return classifier, encoding, weight_sizes, observables

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
