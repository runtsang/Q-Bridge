"""Hybrid quantum classifier that integrates data‑encoding, a variational ansatz, and a quantum self‑attention block."""
from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit


def build_classifier_circuit(num_qubits: int, depth: int,
                             attention_qubits: int = 4) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered quantum circuit that:
        1. Encodes data with RX rotations.
        2. Applies `depth` variational layers of Ry + CZ.
        3. Executes a quantum self‑attention block on the last `attention_qubits` qubits.
    Returns:
        - circuit: QuantumCircuit
        - all_params: list of all Parameter objects (encoding + variational + attention)
        - all_params (duplicate for API compatibility)
        - observables: list of PauliZ operators for measurement.
    """
    # Parameter vectors
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    attn_rot = ParameterVector("alpha", attention_qubits * 3)
    attn_crx = ParameterVector("beta", attention_qubits - 1)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Quantum self‑attention block on the last `attention_qubits` qubits
    start = num_qubits - attention_qubits
    for i, rot in enumerate(attn_rot):
        circuit.rx(rot, start + i)
        circuit.ry(rot, start + i)
        circuit.rz(rot, start + i)
    for i in range(attention_qubits - 1):
        circuit.crx(attn_crx[i], start + i, start + i + 1)

    # Observables: single‑qubit Z on each qubit
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    all_params = list(encoding) + list(weights) + list(attn_rot) + list(attn_crx)
    return circuit, all_params, all_params, observables


class QuantumHybridClassifierQML:
    """
    Wrapper that builds the hybrid circuit and runs it on a backend.
    """
    def __init__(self, num_qubits: int, depth: int, attention_qubits: int = 4):
        self.num_qubits = num_qubits
        self.depth = depth
        self.attention_qubits = attention_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit, self.params, _, self.observables = build_classifier_circuit(
            num_qubits, depth, attention_qubits
        )

    def run(self, input_angles: np.ndarray, shots: int = 1024):
        """
        Execute the circuit for each set of input angles.
        :param input_angles: shape (n_samples, n_params)
        :return: list of measurement count dictionaries per sample
        """
        results = []
        for sample in input_angles:
            binding = dict(zip(self.circuit.parameters, sample))
            bound_circuit = self.circuit.bind_parameters(binding)
            job = qiskit.execute(bound_circuit, self.backend, shots=shots)
            results.append(job.result().get_counts(bound_circuit))
        return results


__all__ = ["build_classifier_circuit", "QuantumHybridClassifierQML"]
