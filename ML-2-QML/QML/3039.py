"""Quantum hybrid model that mirrors the classical HybridNATModel architecture.

The circuit consists of:
1. Data encoding via a 4‑qubit general encoder.
2. A depth‑controlled variational layer that applies rotations and CZ gates.
3. Measurement of all qubits followed by classical batch‑norm.

The design is intentionally aligned with the classical model so that experiments can be performed side‑by‑side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Tuple, List

# Optional quantum circuit factory for benchmarking or visualization
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]
        * quantum circuit
        * list of encoding parameters
        * list of variational parameters
        * list of measurement observables
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class HybridNATModel(tq.QuantumModule):
    """
    Quantum module that implements the hybrid architecture.

    The forward method:
    - Encodes a pooled feature vector into 4 qubits.
    - Applies a variational layer with depth‑controlled rotations and CZ gates.
    - Measures all qubits and normalizes the output.
    """

    class VariationalLayer(tq.QuantumModule):
        def __init__(self, depth: int = 2):
            super().__init__()
            self.depth = depth
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cz = tq.CZ(has_params=False)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for _ in range(self.depth):
                for wire in range(self.n_wires):
                    self.ry(qdev, wire)
                for wire in range(self.n_wires - 1):
                    self.cz(qdev, [wire, wire + 1])
            self.rx(qdev, 0)
            self.rz(qdev, 3)
            self.crx(qdev, [0, 2])
            tqf.hadamard(qdev, 3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, 2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, [3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, depth: int = 2):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.VariationalLayer(depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridNATModel"]
