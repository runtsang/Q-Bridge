"""
SelfAttentionHybrid – quantum variational self‑attention and regression.

The quantum side implements a parameter‑shifted variational circuit
using Pennylane.  The circuit structure mirrors the classical
attention pattern: per‑qubit rotations followed by a chain of
controlled‑RX entanglers.  The `run` method returns expectation
values that can be interpreted as attention scores.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
#  Quantum self‑attention (variational circuit)                               #
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """
    Variational self‑attention block implemented with Pennylane.
    """
    def __init__(self, n_qubits: int = 4, num_layers: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        # Random initial parameters
        self.params = np.random.uniform(0, 2 * np.pi,
                                        size=(num_layers, n_qubits, 3))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: np.ndarray):
            # Input encoding (one‑hot or angle‑encoding could be added)
            for i in range(n_qubits):
                qml.RX(params[0, i, 0], wires=i)
                qml.RY(params[0, i, 1], wires=i)
                qml.RZ(params[0, i, 2], wires=i)

            # Entangling layers
            for layer in range(1, num_layers):
                for i in range(n_qubits - 1):
                    qml.CRX(params[layer, i, 0], wires=(i, i + 1))

            # Return a vector of expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def run(self, inputs: torch.Tensor, params: np.ndarray | None = None) -> torch.Tensor:
        """
        Execute the circuit and return expectation values.

        Parameters
        ----------
        inputs : torch.Tensor
            Placeholder for compatibility – the current circuit does not
            encode input data but the API allows future extensions.
        params : np.ndarray, optional
            Parameters to override the default random initialization.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, n_qubits) containing expectation values.
        """
        if params is None:
            params = self.params
        return self.circuit(inputs, params)


# --------------------------------------------------------------------------- #
#  Quantum regression dataset and model                                       #
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int):
    """
    Generate a quantum state superposition for regression.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that yields complex quantum states and real targets.
    """
    def __init__(self, samples: int, num_wires: int):
        super().__init__()
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(nn.Module):
    """
    Hybrid quantum‑classical regression model.
    The encoder is a Pennylane variational circuit; the head is a
    classical linear layer.
    """
    class QLayer(nn.Module):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.dev = qml.device("default.qubit", wires=num_wires, shots=1024)

            @qml.qnode(self.dev, interface="torch")
            def qnode(inputs: torch.Tensor, params: np.ndarray):
                for i in range(num_wires):
                    qml.RX(params[0, i, 0], wires=i)
                    qml.RY(params[0, i, 1], wires=i)
                    qml.RZ(params[0, i, 2], wires=i)
                for i in range(num_wires - 1):
                    qml.CRX(params[1, i, 0], wires=(i, i + 1))
                return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

            self.qnode = qnode
            self.params = np.random.uniform(0, 2 * np.pi,
                                            size=(2, num_wires, 3))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.qnode(inputs, self.params)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = self.QLayer(num_wires)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        features = self.encoder(state_batch)
        return self.head(features).squeeze(-1)


__all__ = [
    "QuantumSelfAttention",
    "RegressionDataset",
    "QModel",
    "generate_superposition_data",
]
