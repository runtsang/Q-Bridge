"""
QuantumHybridRegression.qml

Hybrid classical‑quantum regression/classification model that uses a
variational circuit built with torchquantum.  The architecture mirrors
the classical counterpart but replaces the dense head with a quantum
expectation layer, and augments the feature extractor with a
parameterised encoder.

The model is fully differentiable thanks to torchquantum’s
autograd support, enabling end‑to‑end training in a single PyTorch
graph.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

__all__ = ["QuantumHybridRegression", "RegressionDataset", "generate_superposition_data"]

# --------------------------------------------------------------------
# Data generation utilities (identical to the classical seed)
# --------------------------------------------------------------------
def generate_superposition_data(num_wires: int,
                                num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic states of the form
    cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Returns states as complex arrays and corresponding labels.
    """
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(num_samples)
    phis = 2 * np.pi * np.random.rand(num_samples)
    states = np.zeros((num_samples, 2 ** num_wires), dtype=complex)
    for i in range(num_samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields a dictionary with ``states`` and ``target``."""

    def __init__(self, num_samples: int, num_wires: int):
        self.states, self.targets = generate_superposition_data(num_wires, num_samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.targets[idx], dtype=torch.float32)}


# --------------------------------------------------------------------
# Hybrid layer that forwards through a variational circuit
# --------------------------------------------------------------------
class QuantumLayer(tq.QuantumModule):
    """
    Parameterised variational circuit that operates on a QuantumDevice.
    The circuit consists of a RandomLayer followed by trainable RX/RY gates.
    """

    def __init__(self, num_wires: int, n_ops: int = 30) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


class HybridHead(tq.QuantumModule):
    """
    Quantum expectation head that returns the mean Pauli‑Z value
    of each qubit.  The output is fed into a classical linear layer.
    """

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.out_dim = num_wires

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        return self.measure(qdev)  # shape (batch, num_wires)


# --------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------
class QuantumHybridRegression(tq.QuantumModule):
    """
    Hybrid classical‑quantum regression/classification model.

    The architecture follows the same scaling strategy used in the classical
    seed: a feature extractor (classical MLP) followed by a variational
    circuit and a quantum expectation head.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    hidden_dims : list[int], optional
        Sizes of the hidden layers in the classical feature extractor.
    regression : bool, default=True
        If True the model outputs a continuous value.
        If False the model outputs binary probabilities.
    shift : float, default=0.0
        Shift value used in the quantum hybrid head (binary classification only).
    num_wires : int, default=3
        Number of qubits used in the variational circuit.
    """

    def __init__(self,
                 num_features: int,
                 hidden_dims: list[int] | None = None,
                 regression: bool = True,
                 shift: float = 0.0,
                 num_wires: int = 3) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        # Classical feature extractor
        layers = []
        in_dim = num_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        self.regression = regression
        self.num_wires = num_wires

        # Quantum part
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.var_circuit = QuantumLayer(num_wires, n_ops=30)
        self.quantum_head = HybridHead(num_wires)

        # Classical head that maps expectation values to final output
        self.head = nn.Linear(num_wires, 1)

        # Shift used for binary classification
        self.shift = shift

    def forward(self, states: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        states : torch.Tensor of shape (batch, num_features)

        Returns
        -------
        torch.Tensor
            For regression: shape (batch,).
            For binary classification: shape (batch, 2) with probabilities.
        """
        # Classical feature extraction
        features = self.feature_extractor(states)

        # Quantum encoding
        bsz = features.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=states.device)
        # Build a state vector from the classical features by normalising per sample.
        target_dim = 2 ** self.num_wires
        state_vectors = torch.nn.functional.normalize(features, dim=1)
        if state_vectors.shape[1] >= target_dim:
            state_vectors = state_vectors[:, :target_dim]
        else:
            pad = torch.zeros(bsz, target_dim - state_vectors.shape[1],
                              dtype=state_vectors.dtype, device=state_vectors.device)
            state_vectors = torch.cat([state_vectors, pad], dim=1)

        self.encoder(qdev, state_vectors)

        # Variational circuit
        self.var_circuit(qdev)

        # Quantum expectation
        q_expectation = self.quantum_head(qdev)  # shape (bsz, num_wires)

        # Classical head
        raw_output = self.head(q_expectation).squeeze(-1)

        if self.regression:
            return raw_output
        else:
            probs = torch.sigmoid(raw_output).squeeze(-1)
            return torch.stack([probs, 1.0 - probs], dim=-1)

    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper for inference.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(states)
