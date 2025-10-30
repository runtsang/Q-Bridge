from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum 2×2 patch encoder using a random circuit."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuantumClassifierModel(tq.QuantumModule):
    """Quantum hybrid classifier: quantum convolution followed by a variational ansatz."""
    def __init__(self, num_qubits: int, depth: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.qfilter = QuanvolutionFilterQuantum()
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = self._build_ansatz()

    def _build_ansatz(self) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        for layer in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(self.weights[layer * self.num_qubits + qubit], qubit)
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantum feature extraction via convolution
        features = self.qfilter(x)
        # Encode features into qubits
        qdev = tq.QuantumDevice(self.num_qubits, bsz=features.shape[0], device=x.device)
        for i in range(features.shape[0]):
            data = features[i]
            for q, val in enumerate(data[:self.num_qubits]):
                qdev.ry(val, q)
        # Apply variational ansatz
        self.circuit.bind_parameters({self.weights[j]: 0.0 for j in range(len(self.weights))})
        qdev.apply(self.circuit)
        # Measure all qubits
        meas = tq.MeasureAll(tq.PauliZ)(qdev)
        return meas

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Build a hybrid quantum classifier circuit that mirrors the classical interface.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (should match number of features after patch encoding).
    depth : int
        Depth of the variational ansatz.

    Returns
    -------
    circuit : QuantumCircuit
        Quantum circuit implementing the hybrid classifier.
    encoding : Iterable
        Parameter vector names for data encoding.
    weights : Iterable
        Parameter vector names for variational parameters.
    observables : list[SparsePauliOp]
        Pauli observables used for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)
    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
