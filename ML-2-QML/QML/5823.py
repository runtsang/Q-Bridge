"""Hybrid regression model using a true quantum circuit, a quantum‑inspired fully connected layer,
and a classical head. The quantum part employs torchquantum for device management
and a random parameterized layer, while the FCL is implemented as a small
classical network to emulate the quantum behaviour in a hybrid fashion."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_hybrid_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data in the quantum domain.
    States are superpositions of |0...0> and |1...1> with random phases.
    Labels follow a trigonometric function of the preparation angles.
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
    return states.astype(np.complex64), labels.astype(np.float32)

class HybridQDataset(torch.utils.data.Dataset):
    """
    Dataset providing quantum states and regression targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_hybrid_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumInspiredFCL(tq.QuantumModule):
    """
    A minimal quantum‑inspired fully connected layer implemented as a
    small classical network that operates on the measurement results.
    This layer mirrors the behaviour of the classical FCL but is placed
    in the quantum branch so that the hybrid model can learn to use it
    in conjunction with the quantum circuit.
    """
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # We assume the caller has already performed measurement and
        # passed the resulting tensor as an attribute on qdev.
        # For simplicity, we take the measurement result directly.
        x = qdev.measure_result  # shape: (batch, n_features)
        values = x.view(-1, 1).float()
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation

class HybridRegression(tq.QuantumModule):
    """
    Quantum‑classical hybrid regression network:
    1. A quantum encoder that maps the input state to a circuit state.
    2. A quantum parameterized layer that entangles the wires.
    3. A measurement that yields a feature vector.
    4. A quantum‑inspired fully connected layer that aggregates the features.
    5. A classical linear head that outputs the scalar prediction.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fcl = QuantumInspiredFCL(num_wires)
        self.head = nn.Linear(1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the input state
        self.encoder(qdev, state_batch)
        # Apply the quantum layer
        self.q_layer(qdev)
        # Measure all qubits to obtain a feature vector
        features = self.measure(qdev)
        # Store measurement result for the FCL
        qdev.measure_result = features
        # Quantum‑inspired fully connected aggregation
        fcl_out = self.fcl(qdev)
        # Final linear head
        return self.head(fcl_out.unsqueeze(-1)).squeeze(-1)

__all__ = ["HybridRegression", "HybridQDataset", "generate_hybrid_data"]
