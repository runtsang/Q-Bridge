"""Quantum classifier with configurable entanglement and noise support.

The circuit encodes the data with RX rotations, then applies a depth‑wise
variational ansatz. Entanglement can be chosen via a string flag.
The class exposes the same API as the classical version.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator

# Optional noise modelling
try:
    from qiskit.providers.aer.noise import NoiseModel
except Exception:
    NoiseModel = None  # pragma: no cover


def _entangle_layer(circuit: QuantumCircuit, entanglement: str) -> None:
    """Add an entanglement pattern to the circuit."""
    n = circuit.num_qubits
    if entanglement == "cnot_ladder":
        for q in range(n - 1):
            circuit.cx(q, q + 1)
    elif entanglement == "cnot_full":
        for q1 in range(n):
            for q2 in range(q1 + 1, n):
                circuit.cx(q1, q2)
    else:
        # default to no entanglement
        pass


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "cnot_ladder",
) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Return a Qiskit circuit that can be used as a classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / input features.
    depth : int
        Depth of the variational layers.
    entanglement : str, optional
        Type of entanglement pattern ('cnot_ladder', 'cnot_full', or None).

    Returns
    -------
    circuit
        The parameterised quantum circuit.
    encoding
        Parameter vector for data encoding.
    weights
        Parameter vector for variational parameters.
    observables
        Pauli Z observables on each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX rotations
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational ansatz
    for layer in range(depth):
        for qubit, param in enumerate(weights[layer * num_qubits : (layer + 1) * num_qubits]):
            circuit.ry(param, qubit)
        _entangle_layer(circuit, entanglement)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding], [weights], observables


class QuantumClassifierModel:
    """Wrapper that evaluates the circuit on a simulator.

    The API matches the classical counterpart so that the two models can be
    swapped in a benchmark pipeline.
    """

    def __init__(self, num_qubits: int, depth: int, entanglement: str = "cnot_ladder"):
        self.num_qubits = num_qubits
        self.depth = depth
        (
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
        ) = build_classifier_circuit(num_qubits, depth, entanglement)
        self.backend = AerSimulator()

    def evaluate(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Return expectation values of the observables.

        Parameters
        ----------
        x
            Data vector of shape (num_qubits,).
        theta
            Variational parameters of shape (num_qubits * depth,).
        """
        param_dict = {}
        for vec, vec_values in zip(self.encoding + self.weights, np.concatenate([x, theta])):
            for param, value in zip(vec, vec_values):
                param_dict[param] = value
        bound = self.circuit.bind_parameters(param_dict)
        job = execute(bound, self.backend, shots=1024)
        result = job.result()
        state = Statevector.from_instruction(bound)
        return np.array([np.real(state.expectation_value(obs)) for obs in self.observables])

    def get_encoding(self) -> Iterable[int]:
        """Return encoding indices (identical to classical API)."""
        return list(range(self.num_qubits))

    def get_observables(self) -> Iterable[SparsePauliOp]:
        """Return the list of observables used."""
        return self.observables


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
