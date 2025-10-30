"""Core circuit factory for the incremental data‑uploading classifier with tunable entanglement and readout."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

def _entanglement_schedule(num_qubits: int, depth: int) -> List[List[Tuple[int, int]]]:
    """
    Generate a list of entanglement pairs for each layer.
    Alternates between a linear chain and a ring pattern.
    """
    schedule: List[List[Tuple[int, int]]] = []
    for d in range(depth):
        if d % 2 == 0:
            # Linear chain: (0,1), (1,2),...
            pairs: List[Tuple[int, int]] = [(i, i + 1) for i in range(num_qubits - 1)]
        else:
            # Ring: linear chain plus (n-1,0)
            pairs = [(i, i + 1) for i in range(num_qubits - 1)] + [(num_qubits - 1, 0)]
        schedule.append(pairs)
    return schedule

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a variational circuit with:
    - data‑encoding via Rx gates,
    - depth‑wise Ry rotations,
    - tunable entanglement schedule (alternating linear/ring),
    - measurement‑based readout (measure_all).

    Returns:
        circuit: QuantumCircuit instance
        encoding: list of ParameterVector objects (identity encoding)
        weights: list of ParameterVector objects (variational parameters)
        observables: list of SparsePauliOp objects for Z measurement on each qubit
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Entanglement schedule
    schedule = _entanglement_schedule(num_qubits, depth)

    for layer_idx in range(depth):
        # Ry rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[layer_idx * num_qubits + qubit], qubit)
        # Entanglement
        for q1, q2 in schedule[layer_idx]:
            circuit.cz(q1, q2)

    # Measurement‑based readout
    circuit.measure_all()

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables
