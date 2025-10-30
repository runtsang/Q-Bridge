"""Quantum regression model using Pennylane with variational circuit and feature extraction."""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as pnp

def generate_superposition_data(num_qubits: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of quantum states |ψ(θ,ϕ)> = cosθ|0…0> + e^{iϕ} sinθ |1…1>
    and corresponding labels f(θ,ϕ) = sin(2θ) cosϕ.
    """
    states = np.zeros((samples, 2**num_qubits), dtype=complex)
    labels = np.empty(samples, dtype=np.float32)
    for i in range(samples):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        omega0 = np.zeros(2**num_qubits, dtype=complex); omega0[0] = 1.0
        omega1 = np.zeros(2**num_qubits, dtype=complex); omega1[-1] = 1.0
        states[i] = np.cos(theta) * omega0 + np.exp(1j*phi) * np.sin(theta) * omega1
        labels[i] = np.sin(2*theta) * np.cos(phi)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset returning quantum states as tensors of dtype torch.cfloat.
    """
    def __init__(self, samples: int, num_qubits: int):
        self.states, self.labels = generate_superposition_data(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionModel(nn.Module):
    """
    Pennylane‑based regression model:
      * Feature encoding via a state‑prep circuit.
      * Variational circuit with entangling layers.
      * Measurement of expectation values of Z on each qubit.
      * Linear head mapping features to a scalar.
    """
    def __init__(self, num_qubits: int, n_layers: int = 3, n_params: int = 6):
        super().__init__()
        self.num_qubits = num_qubits
        self.n_layers = n_layers
        self.n_params = n_params

        # Device for the quantum circuit
        self.dev = qml.device("default.qubit", wires=num_qubits, shots=None)

        # Parameterized variational circuit
        self.var_params = nn.Parameter(torch.randn(n_layers, num_qubits, n_params))

        # Linear head
        self.head = nn.Linear(num_qubits, 1)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(state, params):
            # Load arbitrary state
            qml.StatePrep(state, wires=range(num_qubits))
            # Variational layers
            for l in range(n_layers):
                for q in range(num_qubits):
                    qml.RX(params[l, q, 0], wires=q)
                    qml.RY(params[l, q, 1], wires=q)
                    qml.RZ(params[l, q, 2], wires=q)
                # Entanglement
                for q in range(num_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
            # Measurement of Z expectation values
            return [qml.expval(qml.PauliZ(w)) for w in range(num_qubits)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        state_batch: (batch_size, 2**num_qubits) complex torch tensor
        """
        batch_size = state_batch.shape[0]
        features = self.circuit(state_batch, self.var_params).reshape(batch_size, self.num_qubits)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumRegressionModel", "RegressionDataset", "generate_superposition_data"]
