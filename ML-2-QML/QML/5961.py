"""Hybrid quantum regression model with convolutional feature extraction.

The module extends the original quantum regression seed by adding a
parameter‑free quantum convolution filter (QuanvCircuit) that
preprocesses each sample before it is encoded into the quantum device.
The final regression head receives both the expectation values of the
quantum circuit and the scalar output of the convolution filter,
illustrating how classical quantum‑derived features can be combined with
a variational quantum circuit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# 1. Data generation – same as the original quantum seed
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

# --------------------------------------------------------------------------- #
# 2. Quantum convolution filter – drop‑in replacement for Conv.py
# --------------------------------------------------------------------------- #
def Conv() -> qiskit.QuantumCircuit:
    """Return a quantum filter circuit that acts as a convolutional layer."""
    class QuanvCircuit:
        """Filter circuit used for quanvolution layers."""
        def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            """Run the quantum circuit on classical data."""
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    backend = qiskit.Aer.get_backend("qasm_simulator")
    filter_size = 2
    circuit = QuanvCircuit(filter_size, backend, shots=100, threshold=127)
    return circuit

# --------------------------------------------------------------------------- #
# 3. Dataset – identical to the original quantum seed
# --------------------------------------------------------------------------- #
class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns quantum states and labels."""
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
# 4. Hybrid quantum regression model – includes convolutional feature
# --------------------------------------------------------------------------- #
class HybridRegression(tq.QuantumModule):
    """Quantum regression model that concatenates a quantum convolution filter
    with the main variational circuit.  The final head receives both sets of
    features, illustrating how classical quantum‑derived features can be
    combined with a variational quantum circuit."""
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
        # Classical quantum convolution filter
        self.conv_filter = Conv()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Head now expects num_wires + 1 features (quantum + conv)
        self.head = nn.Linear(num_wires + 1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        # Quantum device for the main circuit
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the raw state
        self.encoder(qdev, state_batch)
        # Variational layer
        self.q_layer(qdev)
        # Quantum features
        quantum_features = self.measure(qdev)  # shape (bsz, n_wires)
        # Classical convolutional features
        conv_features = []
        for sample in state_batch.cpu().numpy():
            # use the real part of the state as classical data
            real_vec = np.real(sample)
            kernel_size = int(np.sqrt(self.n_wires))
            conv_features.append(self.conv_filter.run(real_vec.reshape(kernel_size, kernel_size)))
        conv_tensor = torch.tensor(conv_features, device=state_batch.device, dtype=torch.float32).unsqueeze(-1)
        # Concatenate and feed to head
        combined = torch.cat([quantum_features, conv_tensor], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "Conv"]
