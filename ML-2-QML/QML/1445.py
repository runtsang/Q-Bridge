"""Quantum classifier built with Pennylane.

The circuit uses a data‑encoding layer (RX) followed by a two‑layer
variational ansatz that mixes RY rotations and CZ entanglement.
A parameter‑shift rule is employed for gradient estimation.
The API matches the classical version so the two models can be
plugged into the same experiment harness.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List, Optional
from dataclasses import dataclass

__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]


@dataclass
class QuantumClassifierModel:
    """Data‑encoding + variational circuit for binary classification."""

    num_qubits: int
    depth: int
    dev: qml.Device
    params: torch.Tensor

    def __init__(self, num_qubits: int, depth: int = 2, device_name: str = "default.qubit") -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.dev = qml.device(device_name, wires=num_qubits)

        # Parameter vector: encoding + variational parameters
        self.params = torch.zeros(num_qubits + num_qubits * depth, requires_grad=True)

        # Build the qnode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Data encoding
            for i, w in enumerate(x):
                qml.RX(w, wires=i)

            # Variational layers
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                # CZ entanglement
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Return expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the circuit and return a tensor of expectation values."""
        # Ensure batch dimension
        if x.ndim == 1:
            x = x[None, :]
        expvals = []
        for sample in x:
            exp = self._circuit(sample, self.params)
            expvals.append(exp)
        return torch.stack(expvals)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class labels based on the sign of summed expectations."""
        with torch.no_grad():
            exp = self.forward(x)
            probs = (exp.mean(dim=1) > 0).float()
            return probs.long()

    def evaluate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return raw expectations and probability estimates."""
        exp = self.forward(x)
        probs = (exp.mean(dim=1) > 0).float()
        return exp, probs

    def train_loop(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 30,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Simple gradient‑descent training using the parameter‑shift rule."""
        device = device or torch.device("cpu")
        self.params = self.params.to(device)
        opt = torch.optim.Adam([self.params], lr=lr, weight_decay=weight_decay)

        criterion = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            opt.zero_grad()
            probs = self.forward(x_train)
            logits = probs.mean(dim=1)
            loss = criterion(logits, y_train.float())
            loss.backward()
            opt.step()

    def regularizer_fidelity(self) -> torch.Tensor:
        """Return a fidelity‑based regulariser encouraging the circuit
        to stay close to the identity for unseen inputs.
        """
        samples = torch.randn(10, self.num_qubits)
        fid = 0.0
        for s in samples:
            out = self._circuit(s, self.params)
            fid += out.mean().item()
        return torch.tensor(fid / 10.0)


def build_classifier_circuit(
    num_qubits: int,
    depth: int = 2,
    device_name: str = "default.qubit",
) -> Tuple[QuantumClassifierModel, List[int], List[int], List[qml.PauliZ]]:
    """Factory that mirrors the classical helper interface.

    Returns
    -------
    model : QuantumClassifierModel
        The constructed Pennylane circuit.
    encoding : List[int]
        Indices of wires used for data encoding.
    weight_sizes : List[int]
        Number of trainable parameters per layer.
    observables : List[PauliZ]
        PauliZ observables for each qubit.
    """
    model = QuantumClassifierModel(num_qubits=num_qubits, depth=depth, device_name=device_name)
    encoding = list(range(num_qubits))
    weight_sizes = [model.params.numel()]
    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return model, encoding, weight_sizes, observables
