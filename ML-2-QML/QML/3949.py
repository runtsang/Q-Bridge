"""Hybrid quantum classifier mirroring the classical model and incorporating photonic‑style clipping."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

def build_classifier_circuit(num_qubits: int, depth: int, clip_params: bool = False) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a variational circuit with data‑encoding RX gates and
    photonic‑style variational layers (Ry + CZ).  Parameters can be clipped
    to keep them in a bounded range, mirroring the classical model's
    clipping behaviour.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circ_param = weights[idx]
            if clip_params:
                # Simple clipping to [-π, π] before execution
                circ_param = circ_param.mod(2 * np.pi) - np.pi
            circuit.ry(circ_param, qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

__all__ = ["build_classifier_circuit"]
