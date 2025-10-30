"""Hybrid quantum model that encodes image data, applies a parameterised quantum layer,
and ends with a quantum fully‑connected circuit from the reference pair 2.

The model uses torchquantum for the main circuit but delegates the
parameterised fully‑connected layer to a qiskit circuit in order to
demonstrate cross‑framework interoperability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import execute, Aer
from qiskit.circuit import Parameter
from qiskit.providers import Backend
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumFullyConnectedCircuit:
    """A thin wrapper around the qiskit implementation from reference pair 2."""
    def __init__(self, n_qubits: int = 1, shots: int = 200) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend: Backend = Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("θ")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a list of theta values."""
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys()), dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])


class HybridNATModel(tq.QuantumModule):
    """Quantum‑classical hybrid model that mimics the architecture of the
    original QuantumNAT while appending a quantum fully‑connected circuit
    from the reference pair 2.  The quantum layer operates on the
    flattened image features and the final expectation value is
    returned as a single‑dimensional tensor.
    """
    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer that applies a random circuit followed
        by a small trainable block similar to the original QLayer."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps a 2‑D image to qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        # Quantum fully‑connected circuit from reference pair 2
        self.qfc = QuantumFullyConnectedCircuit(n_qubits=self.n_wires, shots=200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical pooling to match encoder input dimension
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)  # shape (bsz, n_wires)
        out = self.norm(out)

        # Use the first batch element as thetas for the quantum fully‑connected circuit
        thetas = out[0].detach().cpu().numpy().tolist()
        expectation = self.qfc.run(thetas)
        return torch.tensor(expectation, dtype=torch.float32, device=x.device)

__all__ = ["HybridNATModel"]
