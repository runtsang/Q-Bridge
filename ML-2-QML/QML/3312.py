"""Quantum hybrid regression model with a variational circuit and optional Qiskit FCL.

The model encodes classical data into a quantum state, processes it with a
parameterised random layer and single‑qubit rotations, measures all qubits
in the Pauli‑Z basis, and finally maps the expectation values to a scalar
prediction via a classical linear head.  An optional Qiskit circuit can
compute an additional fully‑connected expectation which is concatenated
with the quantum features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from typing import Optional

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic quantum states and labels.

    The states are superpositions of |0...0> and |1...1> with random
    angles, and the labels are a smooth function of the underlying angles.
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
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that yields quantum state tensors and target scalars."""
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
# Optional Qiskit fully connected layer
# --------------------------------------------------------------------------- #
class QiskitFCL:
    """A tiny Qiskit circuit that returns a single expectation value.

    The circuit is parameterised by a single angle and measures
    the expectation of the Pauli‑Z operator.  It is used as a drop‑in
    quantum feature extractor that can be concatenated with the
    torchquantum output.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = Parameter("θ")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for each angle in ``thetas``."""
        jobs = []
        for theta in thetas:
            bound = {self.theta: theta}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bound])
            jobs.append(job)
        results = [j.result() for j in jobs]
        expectations = []
        for res in results:
            counts = res.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            expectations.append(np.sum(states * probs))
        return np.array(expectations, dtype=np.float32).reshape(-1, 1)

# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class QuantumHybridRegression(tq.QuantumModule):
    """Variational regression network with an optional Qiskit FCL feature.

    Parameters
    ----------
    num_wires : int
        Number of qubits used by the variational circuit.
    use_qiskit_fcl : bool, default False
        If True, a QiskitFCL circuit is instantiated and its output
        concatenated with the quantum device measurement before the
        classical head.
    """

    def __init__(self, num_wires: int, use_qiskit_fcl: bool = False):
        super().__init__()
        self.n_wires = num_wires
        # Encoder: a simple Ry rotation per qubit (parameterised by the input)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        # Variational layer: random layer + trainable single‑qubit rotations
        self.q_layer = tq.QuantumModule()
        self.q_layer.add_module("random", tq.RandomLayer(n_ops=30, wires=list(range(num_wires))))
        self.q_layer.add_module("rx", tq.RX(has_params=True, trainable=True))
        self.q_layer.add_module("ry", tq.RY(has_params=True, trainable=True))
        # Measurement: expectation of Pauli‑Z on all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Optional Qiskit FCL
        self.use_qiskit_fcl = use_qiskit_fcl
        if self.use_qiskit_fcl:
            self.fcl = QiskitFCL(n_qubits=1)
        # Classical head
        head_input_dim = num_wires + (1 if self.use_qiskit_fcl else 0)
        self.head = nn.Linear(head_input_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the hybrid quantum regression model."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical data into the quantum state
        self.encoder(qdev, state_batch)
        # Apply variational layer
        self.q_layer(qdev)
        # Measure all qubits
        features = self.measure(qdev)  # shape (bsz, n_wires)
        # Optional Qiskit FCL contribution
        if self.use_qiskit_fcl:
            # Use the first qubit expectation as a proxy angle
            thetas = features[:, 0].cpu().numpy()
            qiskit_expect = self.fcl.run(thetas)  # shape (bsz, 1)
            # Concatenate
            features = torch.cat([features, torch.tensor(qiskit_expect, device=features.device)], dim=1)
        return self.head(features).squeeze(-1)

__all__ = ["QuantumHybridRegression", "RegressionDataset", "generate_superposition_data", "QiskitFCL"]
