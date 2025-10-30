"""Quantum regression module using PennyLane with a hybrid variational circuit.

Key extensions over the seed:
- Encoder uses `qml.QubitStateVector` to load arbitrary state vectors.
- Variational block consists of alternating RY and RZ rotations with entangling CNOTs.
- Supports batch evaluation via PennyLane's `batch` argument.
- Head is a learnable linear layer mapping expectation values to a scalar output.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from typing import Tuple

def generate_superposition_data(num_wires: int, samples: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        omega_0 = np.zeros(2**num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2**num_wires, dtype=complex)
        omega_1[-1] = 1.0
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states.astype(np.complex64), labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset mirroring the classical version but returning complex state vectors.
    """
    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Hybrid quantum‑classical model built with PennyLane and PyTorch.
    """
    def __init__(self, num_wires: int, num_layers: int = 3, n_params_per_layer: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.device = qml.device("default.qubit", wires=num_wires, shots=1024, batch=True)
        # Parameter shape: (num_layers, n_params_per_layer, num_wires)
        self.params = nn.Parameter(torch.randn(num_layers, n_params_per_layer, num_wires))
        # Linear head mapping expectation values to a scalar
        self.head = nn.Linear(num_wires, 1)

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(state: torch.Tensor, params: torch.Tensor):
            # Encode the state vector
            qml.QubitStateVector(state, wires=range(num_wires))
            # Variational layers
            for layer in range(params.shape[0]):
                for wire in range(num_wires):
                    qml.RY(params[layer, 0, wire], wires=wire)
                    qml.RZ(params[layer, 1, wire], wires=wire)
                # Entangling CNOT chain
                for wire in range(num_wires - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            # Measure expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(num_wires)]

        self.circuit = circuit

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex state vectors of shape (batch_size, 2**num_wires).
        Returns
        -------
        torch.Tensor
            Predicted scalar output of shape (batch_size,).
        """
        # PyTorch's complex tensors are supported by PennyLane
        features = self.circuit(state_batch, self.params)
        # Convert list to tensor
        features = torch.stack(features, dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
