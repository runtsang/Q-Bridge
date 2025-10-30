"""Combined quantum regression with a quantum convolution layer.

The quantum model extends the original `QModel` by inserting a
quanvolution circuit before the variational layer.  The output of the quanvolution is concatenated with the variational measurement and fed into a linear head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit.circuit.random import random_circuit

# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class RegressionDataset(torch.utils.data.Dataset):
    """
    Quantum dataset that yields states and targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Quantum convolution circuit (adapted from Conv.py)
# ----------------------------------------------------------------------
class QuanvCircuit:
    """
    Quantum convolution (quanvolution) circuit that measures the
    probability of |1> across all qubits.  The circuit is parameterized
    by the input data and can be executed on a simulator.
    """
    def __init__(self, kernel_size: int, backend=None, shots: int = 100, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """
        Run the circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

# ----------------------------------------------------------------------
# Quantum regression model
# ----------------------------------------------------------------------
class QModel(tq.QuantumModule):
    """
    Quantum regression model that inserts a quanvolution circuit before
    the variational layer.  The output of the quanvolution is concatenated
    with the variational measurement and fed into a linear head.
    """
    class QLayer(tq.QuantumModule):
        """
        Variational layer with random ops and single‑qubit rotations.
        """
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
        # Encoder that maps classical data to a superposition
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational layer
        self.q_layer = self.QLayer(num_wires)
        # Quanvolution circuit
        self.quanv = QuanvCircuit(kernel_size=2, shots=200, threshold=0.5)
        # Measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear head that accepts variational + quanvolution features
        self.head = nn.Linear(num_wires + 1, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Shape (batch, 2**num_wires).  The data is reshaped to a
            2×2 patch for the quanvolution circuit.

        Returns
        -------
        torch.Tensor
            Predicted regression values, shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical data
        self.encoder(qdev, state_batch)
        # Variational layer
        self.q_layer(qdev)
        # Variational measurement
        features = self.measure(qdev)  # (batch, num_wires)

        # Quanvolution measurement (classical post‑processing)
        # Reshape each sample to 2×2 for the quanvolution circuit
        quanv_features = []
        for i in range(bsz):
            sample = state_batch[i].cpu().numpy().reshape(2, 2)
            prob = self.quanv.run(sample)
            quanv_features.append(prob)
        quanv_features = torch.tensor(quanv_features, dtype=torch.float32, device=state_batch.device)
        quanv_features = quanv_features.unsqueeze(-1)  # (batch, 1)

        # Concatenate variational and quanvolution features
        combined = torch.cat([features, quanv_features], dim=1)
        return self.head(combined).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
