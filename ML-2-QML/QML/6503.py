"""Quantum regression model implemented with PennyLane.

The module defines ``QuantumRegression__gen334`` that encodes amplitude‑prepared
states, applies a variational circuit with entanglement, measures Pauli‑Z
expectations, and uses a classical linear head for the final prediction.
"""

import numpy as np
import torch
from torch import nn
import pennylane as qml

def generate_superposition_data(num_wires: int, samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset of quantum states of the form
    cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
    The target is a nonlinear function of the angles.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the state.
    samples : int
        Number of samples to generate.
    seed : int | None
        Optional random seed.

    Returns
    -------
    states : np.ndarray of shape (samples, 2**num_wires)
        Complex amplitude vectors.
    labels : np.ndarray of shape (samples,)
        Continuous target values.
    """
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex128), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary containing the state vectors
    under the key ``states`` and the target under ``target``.
    """
    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegression__gen334(nn.Module):
    """
    Hybrid quantum‑classical regression model that:
    * Encodes the input state using amplitude encoding.
    * Applies a parameterised variational circuit with entanglement.
    * Measures Pauli‑Z expectation values on each qubit.
    * Feeds the measurement results into a linear head.
    """
    def __init__(self, num_wires: int, num_variational_layers: int = 3, device: str | None = None):
        super().__init__()
        self.num_wires = num_wires
        self.num_variational_layers = num_variational_layers
        self.device = device or "default.qubit"

        # PennyLane quantum device – batch mode is enabled automatically
        self.q_device = qml.device(self.device, wires=num_wires, shots=None)

        # Variational parameters – one rotation per qubit per layer
        self.params = nn.Parameter(torch.randn(num_variational_layers, num_wires, 3))

        # Classical head
        self.head = nn.Linear(num_wires, 1)

        # Compile the QNode
        self._qnode = qml.qnode(self._circuit, interface="torch", diff_method="backprop")

    def _circuit(self, state, params):
        # Amplitude encoding
        qml.StatePreparation(state, wires=range(self.num_wires))

        # Variational layers
        for layer in range(self.num_variational_layers):
            for qubit in range(self.num_wires):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RY(params[layer, qubit, 1], wires=qubit)
                qml.RZ(params[layer, qubit, 2], wires=qubit)
            # Entanglement – a simple chain of CNOTs
            for qubit in range(self.num_wires - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        # Measurement – expectation values of Pauli Z on each wire
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor of shape (batch, 2**num_wires)

        Returns
        -------
        output : torch.Tensor of shape (batch,)
        """
        # The QNode automatically handles batch dimension
        features = self._qnode(state_batch, self.params)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegression__gen334", "RegressionDataset", "generate_superposition_data"]
