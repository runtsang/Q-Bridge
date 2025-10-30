"""Quantum regression model built with Pennylane.

The quantum part implements a variational circuit with alternating
parameterized rotations and CNOT entanglement.  States are encoded
via amplitude preparation.  The circuit outputs a vector of
expectation values of Pauli‑Z on each wire, which is fed to a
classical linear head.  This mirrors the transformer‑based classical
model while exposing the benefits of quantum feature extraction.

The dataset generator is identical to the classical one but returns
complex‑valued states for compatibility with Pennylane.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import Dataset
from typing import Tuple

def generate_superposition_data(num_wires: int,
                                samples: int,
                                noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data in the form of complex amplitude
    vectors.  Each sample is a superposition of |0...0⟩ and |1...1⟩
    with random angles.  The target is a smooth function of the angles.

    Parameters
    ----------
    num_wires : int
        Number of qubits / wires.
    samples : int
        Number of samples.
    noise_std : float, optional
        Gaussian noise added to the target.

    Returns
    -------
    states : np.ndarray
        Complex amplitude matrix of shape (samples, 2**num_wires).
    labels : np.ndarray
        Target vector of shape (samples,).
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        # |0...0⟩ component
        state = np.cos(thetas[i]) * np.eye(1, 2 ** num_wires)[0]
        # |1...1⟩ component
        state += np.exp(1j * phis[i]) * np.sin(thetas[i]) * np.eye(1, 2 ** num_wires)[-1]
        states[i] = state
    labels = np.sin(2 * thetas) * np.cos(phis)
    if noise_std > 0.0:
        labels += np.random.normal(0.0, noise_std, size=labels.shape).astype(np.float32)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch dataset that returns complex state vectors and real targets.
    """

    def __init__(self,
                 samples: int,
                 num_wires: int,
                 noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(
            num_wires, samples, noise_std
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(nn.Module):
    """
    Hybrid quantum‑classical regression model using Pennylane.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    n_layers : int, default 2
        Number of variational layers.
    n_params_per_layer : int, default 3
        Number of parameters per qubit per layer (Ry, Rz, Rx).
    """

    def __init__(self,
                 num_wires: int,
                 n_layers: int = 2,
                 n_params_per_layer: int = 3):
        super().__init__()
        self.num_wires = num_wires
        self.n_layers = n_layers
        self.n_params_per_layer = n_params_per_layer

        # PennyLane device with automatic differentiation via Torch
        self.dev = qml.device("default.qubit", wires=num_wires, shots=None)

        # Parameter shape: (n_layers, num_wires, n_params_per_layer)
        self.params = nn.Parameter(
            torch.randn(n_layers, num_wires, n_params_per_layer, dtype=torch.float32)
        )

        # Classical head
        self.head = nn.Linear(num_wires, 1)

    def circuit(self, state: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit that encodes the input state via amplitude preparation
        and applies a variational ansatz.

        Parameters
        ----------
        state : torch.Tensor
            Input state vector of shape (2**num_wires,).
        params : torch.Tensor
            Variational parameters of shape
            (n_layers, num_wires, n_params_per_layer).

        Returns
        -------
        torch.Tensor
            Expectation values of Pauli‑Z on each wire.
        """
        # Prepare the input state
        qml.StatePrep(state, wires=range(self.num_wires))

        # Variational layers
        for layer in range(self.n_layers):
            for wire in range(self.num_wires):
                qml.RY(params[layer, wire, 0], wires=wire)
                qml.RZ(params[layer, wire, 1], wires=wire)
                qml.CNOT(wires=[wire, (wire + 1) % self.num_wires])
                qml.RX(params[layer, wire, 2], wires=wire)

        # Measurement
        return [qml.expval(qml.PauliZ(w)) for w in range(self.num_wires)]

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of state vectors of shape (batch, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Predicted targets of shape (batch,).
        """
        batch_size = state_batch.shape[0]
        # Define a Torch qnode for batched execution
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(state, params):
            return self.circuit(state, params)

        # Broadcast parameters to batch
        params_batch = self.params.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Compute expectation values for each sample
        features = torch.stack(
            [torch.stack(qnode(state_batch[i], params_batch[i])) for i in range(batch_size)]
        )

        # Classical head
        return self.head(features).squeeze(-1)

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
