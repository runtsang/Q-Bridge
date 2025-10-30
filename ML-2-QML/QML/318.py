"""Quantum implementation of the enhanced QuantumNAT model.

Key features:
- 4‑wire variational ansatz with a trainable 2‑layer parameterization.
- Angle‑embedding feature map that encodes the classical input.
- State‑vector output via `run` for diagnostics or further processing.
- Batched execution using Pennylane's QNode with Torch interface.

The class exposes `forward` for expectation‑value predictions and `run` for raw state vectors.
"""

import pennylane as qml
import numpy as np
import torch

class QuantumNATEnhancedQML:
    """Hybrid quantum model with a trainable variational circuit."""

    def __init__(self, n_wires: int = 4, layers: int = 2, device_name: str = "default.qubit") -> None:
        self.n_wires = n_wires
        self.layers = layers
        # Device configured for state‑vector simulation
        self.dev = qml.device(device_name, wires=n_wires, shots=None, backend="statevector")

        # Random initial parameters: shape (layers, wires, 2) for RY & RZ
        self.params = np.random.uniform(0, 2 * np.pi, size=(layers, n_wires, 2))

        # QNode with Torch interface for gradient propagation
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        # Separate QNode to obtain state‑vector
        self.state_qnode = qml.QNode(self._state_circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, params: np.ndarray) -> torch.Tensor:
        """Quantum circuit: feature map + variational ansatz + measurement."""
        # Feature map: angle embedding
        qml.templates.AngleEmbedding(x, wires=range(self.n_wires))
        # Variational ansatz
        for l in range(self.layers):
            for w in range(self.n_wires):
                qml.RY(params[l, w, 0], wires=w)
                qml.RZ(params[l, w, 1], wires=w)
            # Entangling layer (ring topology)
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])
        # Expectation values of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def _state_circuit(self, x: torch.Tensor, params: np.ndarray) -> torch.Tensor:
        """Same circuit as `_circuit` but returns the full state‑vector."""
        qml.templates.AngleEmbedding(x, wires=range(self.n_wires))
        for l in range(self.layers):
            for w in range(self.n_wires):
                qml.RY(params[l, w, 0], wires=w)
                qml.RZ(params[l, w, 1], wires=w)
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])
        return qml.state()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch forward pass returning expectation values."""
        return torch.stack([self.qnode(sample, self.params) for sample in x])

    def run(self, x: torch.Tensor) -> np.ndarray:
        """Return the state‑vector for each input sample."""
        states = [self.state_qnode(sample, self.params) for sample in x]
        return torch.stack(states).detach().cpu().numpy()

    def set_parameters(self, new_params: np.ndarray) -> None:
        """Update trainable parameters."""
        self.params = new_params

__all__ = ["QuantumNATEnhancedQML"]
