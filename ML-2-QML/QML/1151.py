"""Hybrid classical‑quantum model with a learnable encoder and a deeper variational circuit.

The QFCModelV2 mirrors the classical counterpart but replaces the final
classification head with a variational quantum circuit.  The classical encoder
produces a feature vector that is used as parameters for the quantum circuit.
The circuit consists of a parameter‑shared ansatz with multiple layers and
measurements of Pauli‑Z on each wire.  The quantum outputs are projected to
four logits via a linear layer and normalised with batch‑norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml

class QFCModelV2(nn.Module):
    """Hybrid classical‑quantum model with a learnable encoder and a deeper
    variational circuit.

    The architecture is:
    1. A convolutional encoder (identical to the ML counterpart) that maps the
       input image to a 16‑dimensional feature vector.
    2. A variational quantum circuit that uses the feature vector as rotation
       angles for each wire.  The circuit contains `num_layers` repetitions of
       a parameter‑shared ansatz (CNOT‑RY‑CNOT) and is executed on a 4‑wire
       device.
    3. A linear head that maps the four expectation values to four logits.
    4. Batch‑norm over the logits.
    """

    def __init__(self, num_layers: int = 3, device_name: str = "default.qubit") -> None:
        super().__init__()
        self.num_layers = num_layers
        self.device_name = device_name

        # Classical encoder (identical to the ML counterpart)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten_size = 64 * 3 * 3

        # Quantum device
        self.q_device = qml.device(self.device_name, wires=4, shots=None)

        # Parameters for the variational ansatz
        # We use a single parameter per wire per layer, shared across layers
        self.ansatz_params = nn.Parameter(torch.randn(4, self.num_layers))

        # Linear head mapping quantum outputs to logits
        self.linear_head = nn.Linear(4, 4)

        # Batch‑norm on logits
        self.norm = nn.BatchNorm1d(4)

        # Compile the quantum node
        self.qnode = qml.QNode(self._circuit, self.q_device, interface="torch")

    def _circuit(self, params: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        """Quantum circuit used by the QNode."""
        # Encode classical features as rotation angles
        for i in range(4):
            qml.RY(encoded[:, i], wires=i)

        # Variational ansatz with parameter sharing
        for layer in range(self.num_layers):
            for wire in range(4):
                qml.RY(params[wire, layer], wires=wire)
            # Entanglement pattern
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])

        # Measure expectation values of Pauli‑Z on each wire
        return [qml.expval(qml.PauliZ(wire)) for wire in range(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalised logits of shape (batch, 4).
        """
        # Classical encoding
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        # Reduce to 4 features to match the number of wires
        encoded = encoded[:, :4]

        # Quantum forward
        q_out = self.qnode(self.ansatz_params, encoded)

        # Linear head and batch‑norm
        logits = self.linear_head(q_out)
        return self.norm(logits)

__all__ = ["QFCModelV2"]
