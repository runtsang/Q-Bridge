"""Quantum classifier using Pennylane variational circuit.

The class exposes the same static `build_classifier_circuit` that returns
the circuit, parameter vectors, and observables.  Training is done via
gradient descent on expectation values.  The model is fully differentiable
and integrates with PyTorch optimizers.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import torch


class QuantumClassifierModel:
    """
    Variational quantum classifier that mirrors the classical interface.
    The circuit uses an angle‑embedding followed by alternating RX–RZ layers
    and a full‑connection entanglement pattern.  Readout is performed on Z
    observables for each qubit and a classical linear layer maps the
    expectation vector to class logits.
    """
    def __init__(self, num_qubits: int, depth: int = 3, device: str = "default.qubit", shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        self.shots = shots
        self._build_circuit()
        self._build_readout()

    def _build_circuit(self) -> None:
        self.dev = qml.device(self.device, wires=self.num_qubits, shots=self.shots)
        self.qnode = qml.qnode(self.dev, interface="torch")(self._circuit)

    def _circuit(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Variational circuit.  `x` is a batch of features, `theta` are parameters.
        """
        # Data encoding
        for i, wire in enumerate(range(self.num_qubits)):
            qml.RX(x[:, i], wires=wire)
        # Ansatz layers
        idx = 0
        for _ in range(self.depth):
            for wire in range(self.num_qubits):
                qml.RZ(theta[idx], wires=wire)
                idx += 1
            for wire in range(self.num_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
        # Expectation values
        return [qml.expval(qml.PauliZ(wire)) for wire in range(self.num_qubits)]

    def _build_readout(self) -> None:
        # Classical readout layer: linear mapping from qubit expectations to logits
        self.readout = torch.nn.Linear(self.num_qubits, 2)

    def forward(self, X: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: evaluate circuit and apply readout.
        """
        # Expectation values shape: (batch, num_qubits)
        exp_vals = self.qnode(X, theta)
        logits = self.readout(exp_vals)
        return logits

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[object, Iterable, Iterable, List[qml.operation.Operator]]:
        """
        Static helper mirroring the quantum interface.
        Returns a qnode, encoding parameters, variational parameters,
        and a list of PauliZ observables for each qubit.
        """
        dev = qml.device("default.qubit", wires=num_qubits)

        def circuit(x, theta):
            for i, wire in enumerate(range(num_qubits)):
                qml.RX(x[:, i], wires=wire)
            idx = 0
            for _ in range(depth):
                for wire in range(num_qubits):
                    qml.RZ(theta[idx], wires=wire)
                    idx += 1
                for wire in range(num_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_qubits)]

        qnode = qml.qnode(dev, interface="torch")(circuit)
        encoding = torch.arange(num_qubits)  # placeholder indices
        theta_params = torch.arange(num_qubits * depth)  # placeholder
        observables = [qml.PauliZ(i) for i in range(num_qubits)]
        return qnode, encoding, theta_params, observables

    def train_model(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stop_patience: int = 10,
    ) -> List[float]:
        """
        Train the variational parameters using Adam on expectation‑value loss.
        Returns the loss history.
        """
        theta = torch.nn.Parameter(torch.randn(self.num_qubits * self.depth, requires_grad=True))
        optimizer = torch.optim.Adam([theta], lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        best_loss = float("inf")
        patience = 0
        loss_history: List[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.forward(xb, theta)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(loader.dataset)
            loss_history.append(epoch_loss)

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    break
        # Store trained parameters for inference
        self.theta = theta.detach()
        return loss_history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Returns class indices for the input batch.
        """
        with torch.no_grad():
            logits = self.forward(X, self.theta)
            return torch.argmax(logits, dim=1)
