"""Core circuit factory and variational classifier using Pennylane.

The implementation mirrors the classical interface while adding a
parameter‑shift optimiser, data‑re‑uploading, and an entangling layer.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pennylane as qml
import torch


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    device: str = "default.qubit",
    shots: int = 1024,
) -> Tuple[qml.QNode, Iterable, Iterable, list[qml.operation.Operator]]:
    """
    Construct a variational circuit with data encoding, parameter‑shift layers,
    and a Pauli‑Z measurement on each qubit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    device : str, default="default.qubit"
        Pennylane device.
    shots : int, default=1024
        Number of shots for the simulator.

    Returns
    -------
    circuit : qml.QNode
        The compiled quantum circuit.
    encoding : list[str]
        Names of the data‑encoding parameters.
    weights : list[str]
        Names of the variational parameters.
    observables : list[qml.operation.Operator]
        Pauli‑Z operators measured on each qubit.
    """
    dev = qml.device(device, wires=num_qubits, shots=shots)

    encoding = [f"x_{i}" for i in range(num_qubits)]
    weights = [f"theta_{i}" for i in range(num_qubits * depth)]

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Data encoding
        for i, wire in enumerate(range(num_qubits)):
            qml.RX(inputs[wire], wires=wire)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for wire in range(num_qubits):
                qml.RY(params[idx], wires=wire)
                idx += 1
            # Entangling layer
            for wire in range(num_qubits - 1):
                qml.CZ(wires=[wire, wire + 1])

        # Measurement
        return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

    observables = [qml.PauliZ(i) for i in range(num_qubits)]
    return circuit, encoding, weights, observables


class QuantumClassifier:
    """Quantum variational classifier implemented with Pennylane.

    The class exposes a training loop that optimises the variational parameters
    with the parameter‑shift rule.  It mirrors the classical interface so that
    both models can be swapped in a research pipeline.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        lr: float = 0.01,
        device: str = "default.qubit",
        shots: int = 1024,
    ):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth, device, shots
        )
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        self.optimizer = qml.AdamOptimizer(stepsize=lr)
        # Initialise parameters
        self.params = torch.tensor(
            0.01 * np.random.randn(num_qubits * depth),
            requires_grad=True,
            dtype=torch.float64,
        )

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross‑entropy loss between the first two qubit expectation values
        (treated as logits) and the true labels.
        """
        logits = logits[:, :2]  # use first two qubits as logits
        return torch.nn.functional.cross_entropy(logits, labels)

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int = 20,
        verbose: bool = False,
    ) -> list[float]:
        """
        Train the variational circuit for a fixed number of epochs.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader yielding ``(X, y)`` batches where ``X`` is a tensor of
            shape ``(N, num_qubits)``.
        epochs : int, default=20
            Number of training epochs.
        verbose : bool, default=False
            Whether to print epoch‑wise loss.

        Returns
        -------
        history : list[float]
            List of average training loss per epoch.
        """
        history: list[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                def cost(p):
                    logits = self.circuit(X, p)
                    return self.loss_fn(logits, y)

                self.params, loss = self.optimizer.step(cost, self.params)
                epoch_loss += loss.item() * X.size(0)

            epoch_loss /= len(dataloader.dataset)
            history.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")
        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class probabilities for the given inputs.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape ``(N, num_qubits)``.

        Returns
        -------
        probs : torch.Tensor
            Probabilities of shape ``(N, 2)``.
        """
        self.circuit.set_options(device=self.device)
        with torch.no_grad():
            logits = self.circuit(X, self.params)
            probs = torch.softmax(logits[:, :2], dim=-1)
        return probs


__all__ = ["build_classifier_circuit", "QuantumClassifier"]
