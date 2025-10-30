"""Hybrid quantum classifier combining quanvolution, sampler QNN and fully connected layer.

The `build_classifier_circuit` function returns a Qiskit QuantumCircuit together with
encoding metadata, weight sizes, and observable definitions.  The design supports
optional inclusion of a quantum quanvolution filter, a sampler QNN, and a fully‑connected
quantum layer, enabling a smooth transition from purely classical to fully quantum
workflows.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumQuanvolutionFilter:
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.encoder_params = ParameterVector("e", n_wires)
        self.random_params = ParameterVector("r", n_wires * 8)
        self.circuit = QuantumCircuit(n_wires)
        for i, p in enumerate(self.encoder_params):
            self.circuit.ry(p, i)
        for i, p in enumerate(self.random_params):
            self.circuit.ry(p, i % n_wires)
        for i in range(n_wires - 1):
            self.circuit.cz(i, i + 1)

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit


class QuantumSamplerQNN:
    """Simple parameterized quantum sampler for 2 qubits."""
    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        self.input_params = ParameterVector("x", num_qubits)
        self.weight_params = ParameterVector("w", num_qubits * 3)
        self.circuit = QuantumCircuit(num_qubits)
        for i, p in enumerate(self.input_params):
            self.circuit.ry(p, i)
        for i, p in enumerate(self.weight_params):
            self.circuit.ry(p, i % num_qubits)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit


class QuantumFCL:
    """Fully connected quantum layer using a single qubit."""
    def __init__(self, num_qubits: int = 1) -> None:
        self.num_qubits = num_qubits
        self.param = ParameterVector("theta", num_qubits)
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.h(range(num_qubits))
        for i, p in enumerate(self.param):
            self.circuit.ry(p, i)
        self.circuit.measure_all()

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit


class QuantumClassifierModelGen:
    """Hybrid quantum classifier that can optionally include quanvolution, sampler QNN, and FCL."""
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 use_quanvolution: bool = True,
                 use_sampler: bool = True,
                 use_fcl: bool = True) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_quanvolution = use_quanvolution
        self.use_sampler = use_sampler
        self.use_fcl = use_fcl
        self.circuit = self._build_circuit()
        self.encoding = list(range(num_qubits))
        self.weight_sizes = self._compute_weight_sizes()
        self.observables = [SparsePauliOp("Z" * num_qubits)]

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # encoding: simple Rx for each qubit
        for i in range(self.num_qubits):
            qc.rx(ParameterVector(f"phi_{i}", 1)[0], i)
        # optional modules
        if self.use_quanvolution:
            quanv = QuantumQuanvolutionFilter()
            qc.append(quanv.get_circuit(), range(self.num_qubits))
        if self.use_sampler:
            sampler = QuantumSamplerQNN()
            qc.append(sampler.get_circuit(), range(self.num_qubits))
        if self.use_fcl:
            fcl = QuantumFCL()
            qc.append(fcl.get_circuit(), range(self.num_qubits))
        qc.measure_all()
        return qc

    def _compute_weight_sizes(self) -> list[int]:
        sizes: list[int] = []
        if self.use_quanvolution:
            sizes.append(4 + 4 * 8)  # n_wires + random ops
        if self.use_sampler:
            sizes.append(2 + 2 * 3)
        if self.use_fcl:
            sizes.append(1)
        return sizes


def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             use_quanvolution: bool = True,
                             use_sampler: bool = True,
                             use_fcl: bool = True) -> Tuple[QuantumCircuit, Iterable[int], Iterable[int], list[SparsePauliOp]]:
    """Return a quantum circuit and metadata for the hybrid classifier."""
    model = QuantumClassifierModelGen(num_qubits, depth,
                                      use_quanvolution, use_sampler, use_fcl)
    return model.circuit, model.encoding, model.weight_sizes, model.observables


__all__ = [
    "build_classifier_circuit",
    "QuantumClassifierModelGen",
    "QuantumQuanvolutionFilter",
    "QuantumSamplerQNN",
    "QuantumFCL",
]
