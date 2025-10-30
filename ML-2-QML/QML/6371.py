"""Hybrid convolutional and regression framework – quantum implementation.

This module implements the same `HybridConvRegression` interface as the
classical version but replaces the convolution filter with a quantum
quanvolution circuit and the regression head with a torchquantum
variational model.  It relies on Qiskit for the circuit and a
fully differentiable quantum backend for training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# Dataset generation (identical to the quantum reference)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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
    """Dataset wrapping the quantum superposition states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Quantum convolution filter (quanvolution)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum filter that processes a 2‑D patch and outputs a probability."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127.0):
        n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on a single kernel patch."""
        data = np.reshape(data, (1, self._circuit.num_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return counts / (self.shots * self._circuit.num_qubits)

# --------------------------------------------------------------------------- #
# Quantum regression model
# --------------------------------------------------------------------------- #
class QModel(tq.QuantumModule):
    """Quantum‑classical hybrid regression model."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
# Unified interface
# --------------------------------------------------------------------------- #
class HybridConvRegression:
    """
    Unified class exposing quantum convolution and regression capabilities.

    Parameters
    ----------
    mode : str, default="quantum"
        Must be 'quantum' to instantiate the quantum components.
    """
    def __init__(self, mode: str = "quantum", **kwargs) -> None:
        if mode!= "quantum":
            raise ValueError("Only quantum mode is supported in this module.")
        self.conv = QuanvCircuit(**kwargs)
        self.regressor = QModel(kwargs.get("num_wires", 4))

    def fit(self, dataset: torch.utils.data.Dataset, epochs: int = 10, lr: float = 1e-3) -> None:
        """Train the quantum regression head on quantum state data."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.regressor.train()
        for _ in range(epochs):
            for batch in loader:
                preds = self.regressor(batch["states"])
                loss = loss_fn(preds, batch["target"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the quantum regression to a batch of raw quantum states."""
        with torch.no_grad():
            return self.regressor(data)

__all__ = ["HybridConvRegression", "RegressionDataset", "generate_superposition_data"]
