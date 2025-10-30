"""Hybrid classical‑quantum classifier using PyTorch and TorchQuantum.

The class exposes a purely classical interface (torch.nn.Module) while internally
leveraging a quantum feature extractor built with TorchQuantum.  The design mirrors
the original `QuantumClassifierModel.py` but augments it with a quantum layer
that can be toggled on or off.  This allows seamless experimentation between
classical, hybrid, and quantum‑only regimes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq


class HybridClassifier(nn.Module):
    """Hybrid classifier with optional quantum feature extraction.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int, default=2
        Number of variational layers in the quantum circuit.
    use_quantum : bool, default=True
        If ``True`` the quantum feature extractor is active.
    latent_dim : int, default=8
        Size of the latent representation produced by both classical
        and quantum branches before fusion.
    """

    def __init__(
        self,
        num_features: int,
        depth: int = 2,
        use_quantum: bool = True,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.use_quantum = use_quantum

        # Classical encoder ------------------------------------------------
        encoder_layers = [
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum feature extractor ---------------------------------------
        if use_quantum:
            self.qmodule = _QuantumFeatureExtractor(
                num_wires=num_features, depth=depth, latent_dim=latent_dim
            )
        else:
            self.qmodule = None

        # Final classifier -------------------------------------------------
        self.classifier = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        The input is first processed by the classical encoder.  If the quantum
        branch is enabled, its output is added element‑wise to the classical
        latent vector before classification.
        """
        latent = self.encoder(x)
        if self.use_quantum:
            q_latent = self.qmodule(x)
            latent = latent + q_latent  # simple fusion
        return self.classifier(latent)


class _QuantumFeatureExtractor(tq.QuantumModule):
    """Encodes the input into a quantum state and applies a variational ansatz.

    The circuit consists of a data‑encoding layer (RX rotations) followed by
    ``depth`` blocks of RX rotations and CZ entangling gates.  The expectation
    values of Pauli‑Z on each qubit are measured and linearly projected to
    ``latent_dim`` dimensions.
    """

    def __init__(
        self,
        num_wires: int,
        depth: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.variational = tq.RandomLayer(
            n_ops=depth * num_wires, wires=list(range(num_wires))
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, latent_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=state_batch.device
        )
        self.encoder(qdev, state_batch)
        self.variational(qdev)
        features = self.measure(qdev)
        return self.head(features)


__all__ = ["HybridClassifier"]
