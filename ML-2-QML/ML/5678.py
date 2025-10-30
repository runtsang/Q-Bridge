"""Hybrid classifier that fuses classical and quantum building blocks.

The module exposes a :class:`QuantumHybridClassifier` that accepts a
``mode`` keyword to switch between:
  * ``"classic"`` – a pure feed‑forward neural net built from PyTorch.
  * ``"quantum"`` – a variational circuit built with Qiskit that encodes
    the input features.
  * ``"hybrid"`` – a hybrid network where the classical backbone
    produces a feature vector that is then fed into a quantum
    classifier.  The quantum part is evaluated via a helper
    function defined in :mod:`qml_module`.

The design is inspired by the two reference pairs:
  * The classical classifier factory (seed 1) provides a clean
    construction of a linear stack.
  * The quantum circuit factory (seed 1) supplies a data‑encoding
    scheme and a variational ansatz.
  * The QLSTM (seed 2) shows how to embed quantum gates inside
    recurrent layers, which we reuse as a *QuantumGateLayer* for
    feature extraction before the quantum classifier.

The class is fully importable and can be used as a drop‑in
replacement in existing pipelines.  It also exposes helper
functions ``build_classical_backbone`` and ``build_gate_layer``
that mirror the original API.

"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the quantum helper module.  It must be present in the same
# package or in the PYTHONPATH.  The quantum code is completely
# isolated from the classical code and only provides helper
# functions for building and evaluating circuits.
try:
    from.qml_module import (
        build_classifier_circuit,
        evaluate_classifier,
    )
except Exception:  # pragma: no cover
    # The quantum module might not be available in a purely classical
    # environment.  In that case the quantum and hybrid modes will
    # raise informative errors when used.
    build_classifier_circuit = None
    evaluate_classifier = None


# --------------------------------------------------------------------------- #
# Classical helper – feed‑forward backbone
# --------------------------------------------------------------------------- #
def build_classical_backbone(
    num_features: int,
    depth: int,
    hidden_dim: int = 32,
) -> Tuple[nn.Module, List[int]]:
    """Return a simple feed‑forward network and its weight‑size list.

    The backbone is built from the same structure used in the
    seed‑one classical factory: each layer maps
    (in‑dim) → (out‑dim) with ReLU activations.  The
    ``weight_sizes`` list is useful for parameter‑budget
    and scaling analysis.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.extend([linear.weight.numel(), linear.bias.numel()])
        in_dim = hidden_dim

    # Final head: binary classification
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.extend([head.weight.numel(), head.bias.numel()])

    network = nn.Sequential(*layers)
    return network, weight_sizes


# --------------------------------------------------------------------------- #
# Hybrid gate layer (re‑used from QLSTM)
# --------------------------------------------------------------------------- #
class QuantumGateLayer(nn.Module):
    """Small quantum gate layer that can be embedded in a classical network.

    The layer is a thin wrapper around a Qiskit circuit that can be
    called from PyTorch.  It is not differentiable in this
    implementation but serves as a feature extractor that can be
    swapped with a differentiable backend if needed.
    """

    def __init__(self, num_qubits: int, depth: int):
        super().__init__()
        if build_classifier_circuit is None:
            raise RuntimeError(
                "Quantum helper module not available; cannot instantiate QuantumGateLayer."
            )
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        # Trainable weight parameters
        self.weight_params = nn.Parameter(
            torch.randn(num_qubits * depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the quantum circuit for each sample in the batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_qubits).  The first
            ``num_qubits`` values are bound to the encoding gates.
            The remaining parameters are the variational weights.
        """
        if evaluate_classifier is None:
            raise RuntimeError(
                "Quantum helper module not available; cannot evaluate quantum circuit."
            )
        # Concatenate encoding and weight parameters
        batch_size = x.size(0)
        encoding = x[:, : self.circuit.num_qubits]
        # Ensure weight_params are expanded to the batch size
        weight = self.weight_params.expand(batch_size, -1)
        params = torch.cat([encoding, weight], dim=1)
        return evaluate_classifier(self.circuit, params, self.observables, shots=1024)


# --------------------------------------------------------------------------- #
# Main hybrid classifier
# --------------------------------------------------------------------------- #
class QuantumHybridClassifier(nn.Module):
    """Drop‑in classifier that can operate in classic, quantum, or hybrid mode.

    Parameters
    ----------
    num_features : int
        Number of input features for the classical backbone.
    num_qubits : int
        Number of qubits used by the quantum circuit.
    depth : int, default 2
        Depth of the variational ansatz.
    mode : str, default "classic"
        One of ``"classic"``, ``"quantum"``, or ``"hybrid"``.
    """

    def __init__(
        self,
        num_features: int,
        num_qubits: int,
        depth: int = 2,
        mode: str = "classic",
    ):
        super().__init__()
        self.mode = mode.lower()
        self.num_features = num_features
        self.num_qubits = num_qubits
        self.depth = depth

        if self.mode not in {"classic", "quantum", "hybrid"}:
            raise ValueError(f"Unsupported mode: {mode!r}")

        if self.mode in {"classic", "hybrid"}:
            self.backbone, self.backbone_weights = build_classical_backbone(
                num_features, depth
            )

        if self.mode in {"quantum", "hybrid"}:
            if build_classifier_circuit is None:
                raise RuntimeError(
                    "Quantum helper module not available; cannot construct quantum circuit."
                )
            self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
                num_qubits, depth
            )
            # Trainable quantum weights
            self.weight_params = nn.Parameter(
                torch.randn(num_qubits * depth)
            )

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of inputs.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, num_features).
        """
        if self.mode == "classic":
            return self.backbone(x)

        if self.mode == "quantum":
            # Use the first ``num_qubits`` features as encoding parameters
            encoding = x[:, : self.num_qubits]
            # Expand weight parameters to batch size
            batch_size = x.size(0)
            weight = self.weight_params.expand(batch_size, -1)
            params = torch.cat([encoding, weight], dim=1)
            return evaluate_classifier(
                self.circuit, params, self.observables, shots=1024
            )

        # Hybrid mode
        features = self.backbone(x)
        # Use the first ``num_qubits`` of the backbone output as encoding
        encoding = features[:, : self.num_qubits]
        batch_size = features.size(0)
        weight = self.weight_params.expand(batch_size, -1)
        params = torch.cat([encoding, weight], dim=1)
        return evaluate_classifier(
            self.circuit, params, self.observables, shots=1024
        )

    # --------------------------------------------------------------------- #
    # Utility methods
    # --------------------------------------------------------------------- #
    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        return total

    def __repr__(self) -> str:
        mode_info = f"mode={self.mode!r}"
        param_info = f"params={self.num_parameters()}"
        return f"<{self.__class__.__name__} {mode_info} {param_info}>"

__all__ = [
    "QuantumHybridClassifier",
    "QuantumGateLayer",
    "build_classical_backbone",
]
