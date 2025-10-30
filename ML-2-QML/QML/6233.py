"""Quantum estimator with a two‑qubit parameter‑shift variational circuit.

The circuit encodes inputs with RY rotations, applies a depth‑controlled variational
layer, and measures Pauli‑Z on each qubit. A simple gradient‑descent trainer is
included for quick experimentation.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch.utils.data import DataLoader


class EstimatorQNN:
    """Two‑qubit variational circuit with parameter‑shift gradients.

    Parameters
    ----------
    dev : pennylane.Device, optional
        Quantum device; defaults to `default.qubit` with 2 wires.
    num_layers : int, default 2
        Depth of the variational layer.
    init_params : np.ndarray, optional
        Initial weights of shape (num_layers, num_qubits, 3). Randomly generated
        if None.
    """

    def __init__(
        self,
        dev: qml.Device | None = None,
        num_layers: int = 2,
        init_params: np.ndarray | None = None,
    ) -> None:
        self.dev = dev or qml.device("default.qubit", wires=2)
        self.num_layers = num_layers
        self.num_qubits = self.dev.num_wires
        self.params = (
            init_params
            if init_params is not None
            else np.random.randn(num_layers, self.num_qubits, 3)
        )
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
            # Input encoding
            for i, w in enumerate(inputs):
                qml.RY(w, wires=i)

            # Variational layers
            for l in range(self.num_layers):
                for q in range(self.num_qubits):
                    qml.RZ(weights[l, q, 0], wires=q)
                    qml.RY(weights[l, q, 1], wires=q)
                    qml.RZ(weights[l, q, 2], wires=q)
                # Entanglement
                for q in range(self.num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

            # Return expectation values of Pauli‑Z on each qubit
            return torch.stack([qml.expval(qml.PauliZ(q)) for q in range(self.num_qubits)])

        self.circuit = circuit

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the circuit for a single batch of inputs."""
        return self.circuit(inputs, self.params)

    def train(
        self,
        dataloader: DataLoader,
        *,
        epochs: int = 100,
        lr: float = 0.01,
        device: str = "cpu",
    ) -> None:
        """Train the circuit using a simple gradient‑descent optimiser.

        Parameters
        ----------
        dataloader : DataLoader
            Iterable over (inputs, targets) pairs.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        device : str
            Device for torch tensors.
        """
        optimizer = qml.GradientDescentOptimizer(lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for x, y in dataloader:
                x = torch.tensor(x, dtype=torch.float32).to(device)
                y = torch.tensor(y, dtype=torch.float32).to(device)

                def loss_fn(params: np.ndarray) -> torch.Tensor:
                    preds = self.circuit(x, params)
                    return ((preds - y) ** 2).mean()

                self.params = optimizer.step(loss_fn, self.params)
                epoch_loss += loss_fn(self.params).item() * x.size(0)

            epoch_loss /= len(dataloader.dataset)
            print(f"Epoch {epoch + 1}/{epochs} – loss: {epoch_loss:.4f}")

    def __repr__(self) -> str:
        return f"<EstimatorQNN layers={self.num_layers} qubits={self.num_qubits}>"


def EstimatorQNN(**kwargs) -> EstimatorQNN:
    """Convenience factory returning a configured EstimatorQNN."""
    return EstimatorQNN(**kwargs)


__all__ = ["EstimatorQNN", "EstimatorQNN"]  # two exports for backward compatibility
