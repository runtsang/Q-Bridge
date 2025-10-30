"""Quantum variant of QFCModel using Pennylane with a learnable variational circuit.

The model accepts a 4‑channel image, encodes each channel into a rotational
gate, applies a depth‑wise trainable rotation layer, entangles adjacent qubits,
and measures the expectation of Pauli‑Z on each wire.  The resulting classical
vector is passed through a linear layer to produce 4 logits.

The circuit is fully differentiable and can be trained end‑to‑end jointly with
a classical backbone or as a standalone module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml


class QFCModel(nn.Module):
    """Variational quantum circuit for 4‑class classification with 4 wires."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3, device: str = "cpu"):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.device = device

        # Trainable parameters for the variational layer
        self.params = nn.Parameter(
            torch.randn(n_layers, n_wires, 3, dtype=torch.float64)
        )

        # Pennylane device (default qubit simulator)
        self.pennylane_dev = qml.device("default.qubit", wires=n_wires)

        # QNode definition
        self.qnode = qml.QNode(self._circuit, self.pennylane_dev,
                               interface="torch", diff_method="backprop")

        # Post‑circuit linear head
        self.linear = nn.Linear(n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> list[torch.Tensor]:
        """Variational circuit applied to the input vector ``x``."""
        for i in range(self.n_layers):
            for w in range(self.n_wires):
                # Encode classical information
                qml.RY(x[w], wires=w)
                # Trainable rotations
                qml.RZ(params[i, w, 0], wires=w)
                qml.RX(params[i, w, 1], wires=w)
                # Entangle with next qubit
                if w < self.n_wires - 1:
                    qml.CNOT(wires=[w, w + 1])
        # Measure expectation values of Pauli‑Z on each wire
        return [qml.expval(qml.PauliZ(wires=w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode image to 4 features, run variational circuit,
        and produce 4 logits."""
        bsz = x.shape[0]
        # Flatten spatial dimensions and group into 4 channels
        flat = x.view(bsz, -1)
        groups = flat.chunk(4, dim=1)
        enc = torch.stack([g.mean(dim=1) for g in groups], dim=1)  # (bsz, 4)

        # Run circuit for each batch element
        out_list = []
        for i in range(bsz):
            out_list.append(self.qnode(enc[i], self.params))
        out = torch.stack(out_list, dim=0)  # (bsz, 4)
        out = self.linear(out)
        return self.norm(out)


__all__ = ["QFCModel"]
