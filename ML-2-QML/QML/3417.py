"""Hybrid quantum regression module.

This module implements a variational quantum circuit that encodes the input
superposition state, applies a trainable parameterised layer, and measures
a Pauli‑Z expectation value.  The output can be used as a feature vector
in a hybrid classical‑quantum workflow.

The design draws from the `QuantumRegression.py` reference, extending it
with a flexible QNode that supports batching, a simple random layer and
parameterised rotations.  It is fully compatible with Torch via
PennyLane's ``interface="torch"``.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pennylane as qml

def generate_superposition_data(num_qubits: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex amplitude‑encoded states and a target.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (log₂ of the state dimensionality).
    samples : int
        Number of samples to generate.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape (samples, 2**num_qubits).
    labels : np.ndarray
        Scalar targets computed as ``sin(2θ) * cos(φ)``.
    """
    dim = 2 ** num_qubits
    omega_0 = np.zeros(dim, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(dim, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, dim), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Torch dataset returning amplitude‑encoded complex states and target."""
    def __init__(self, samples: int, num_qubits: int):
        self.states, self.labels = generate_superposition_data(num_qubits, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class QuantumRegressionCircuit:
    """Variational circuit that encodes the input state and outputs a Pauli‑Z expectation."""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.wires = list(range(num_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires, shots=None)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, thetas):
            # Amplitude encoding of the complex state
            qml.StatePreparation(inputs, wires=self.wires)
            # Parameterised rotation layer
            for i, wire in enumerate(self.wires):
                qml.RX(thetas[i], wires=wire)
                qml.RY(thetas[i], wires=wire)
            # Simple entangling pattern
            for i in range(self.num_qubits - 1):
                qml.CNOT(self.wires[i], self.wires[i + 1])
            # Observation
            return qml.expval(qml.PauliZ(self.wires[0]))
        self.circuit = circuit

    def run(self, batch_inputs: torch.Tensor, batch_thetas: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit for a batch of inputs.

        Parameters
        ----------
        batch_inputs : torch.Tensor
            Complex vectors of shape (batch, 2**num_qubits).
        batch_thetas : torch.Tensor
            Parameters of shape (batch, num_qubits).

        Returns
        -------
        expectations : torch.Tensor
            Pauli‑Z expectation values, shape (batch,).
        """
        expectations = []
        for inp, th in zip(batch_inputs, batch_thetas):
            expectations.append(self.circuit(inp, th))
        return torch.stack(expectations).squeeze(-1)

class HybridFCLRegression(nn.Module):
    """Quantum‑only regression head with trainable parameters."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.circuit = QuantumRegressionCircuit(num_qubits)
        # Initialise trainable rotation parameters
        self.thetas = nn.Parameter(torch.randn(num_qubits))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Return the quantum expectation for a batch of states."""
        batch_size = states.shape[0]
        thetas = self.thetas.unsqueeze(0).expand(batch_size, -1)
        return self.circuit.run(states, thetas)

__all__ = ["HybridFCLRegression", "RegressionDataset", "generate_superposition_data"]
