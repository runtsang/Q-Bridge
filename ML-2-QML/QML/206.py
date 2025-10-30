"""Enhanced quantum classifier factory with configurable encoding and entanglement."""
from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def _build_entangling_layer(circuit: QuantumCircuit,
                            qubits: List[int],
                            pattern: str = "cnot",
                            depth: int = 1) -> None:
    """Apply an entangling pattern across the qubits."""
    if pattern == "cnot":
        for _ in range(depth):
            for i in range(len(qubits) - 1):
                circuit.cx(qubits[i], qubits[i + 1])
    elif pattern == "full":
        for _ in range(depth):
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    circuit.cx(qubits[i], qubits[j])
    elif pattern == "chain":
        for _ in range(depth):
            for i in range(len(qubits) - 1):
                circuit.cx(qubits[i], qubits[i + 1])
    else:
        raise ValueError(f"Unsupported entanglement pattern: {pattern}")


def build_classifier_circuit(num_qubits: int,
                             depth: int,
                             encoding: str = "rx",
                             entanglement: str = "cnot",
                             obs_y: bool = False) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a variational quantum circuit for classification.

    Parameters
    ----------
    num_qubits: int
        Number of qubits in the circuit.
    depth: int
        Number of variational layers.
    encoding: str, optional
        Data‑encoding gate: 'rx', 'ry', 'rz', or 'amplitude'.
    entanglement: str, optional
        Entanglement pattern: 'cnot', 'full', or 'chain'.
    obs_y: bool, optional
        Whether to include Y‑basis observables in addition to Z.

    Returns
    -------
    circuit: QuantumCircuit
        Assembled quantum circuit.
    encoding_params: Iterable
        Parameter vector for data encoding.
    variational_params: Iterable
        Parameter vector for trainable rotations.
    observables: List[SparsePauliOp]
        Pauli operators whose expectation values are returned as logits.
    """
    # Parameters for encoding and variational layers
    if encoding == "amplitude":
        # amplitude encoding uses a single parameter vector per qubit
        encoding_params = ParameterVector("x", num_qubits * depth)
    else:
        encoding_params = ParameterVector("x", num_qubits)

    variational_params = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    if encoding == "rx":
        for q, p in zip(range(num_qubits), encoding_params):
            circuit.rx(p, q)
    elif encoding == "ry":
        for q, p in zip(range(num_qubits), encoding_params):
            circuit.ry(p, q)
    elif encoding == "rz":
        for q, p in zip(range(num_qubits), encoding_params):
            circuit.rz(p, q)
    elif encoding == "amplitude":
        # amplitude encoding via state preparation (placeholder)
        for layer in range(depth):
            for q in range(num_qubits):
                circuit.initialize(encoding_params[layer * num_qubits + q], q)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")

    # Variational layers
    idx = 0
    for _ in range(depth):
        # Single‑qubit rotations
        for q in range(num_qubits):
            circuit.ry(variational_params[idx], q)
            idx += 1
        # Entanglement
        _build_entangling_layer(circuit, list(range(num_qubits)), entanglement, depth=1)

    # Observables: Z on each qubit, optionally Y
    observables: List[SparsePauliOp] = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    if obs_y:
        observables += [
            SparsePauliOp("I" * i + "Y" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

    return circuit, list(encoding_params), list(variational_params), observables


__all__ = ["build_classifier_circuit"]
