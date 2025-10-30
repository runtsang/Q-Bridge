"""Hybrid quantum classifier that mirrors the classical implementation.

The circuit uses a parameterized ansatz with data‑encoding via Rx gates,
followed by alternating Ry layers and entangling CZ gates.  The
parameter vector is split into an encoding part and a variational part
exactly as in the seed.  The `run` method executes the circuit on a
simulator and returns the expectation values of Z observables, which
can be interpreted as logits for a binary classification task.

The module is intentionally lightweight so it can be swapped into
benchmarks or combined with the classical version for hybrid training.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


class HybridClassifier:
    """
    Quantum implementation of the classifier.

    Parameters
    ----------
    num_qubits: int
        Number of qubits; should match the dimensionality of the input.
    depth: int
        Number of variational layers.
    """

    def __init__(self, num_qubits: int, depth: int) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

        # Backend for simulation
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Construct the layered ansatz with data encoding and variational parameters."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)

        # Data encoding (Rx)
        for qubit in range(self.num_qubits):
            qc.rx(encoding[qubit], qubit)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            # Entangling pattern: nearest‑neighbour CZ
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Measurement of all qubits
        qc.measure_all()

        # Observables: single‑qubit Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
                       for i in range(self.num_qubits)]

        return qc, list(encoding), list(weights), observables

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the given variational parameters.

        Parameters
        ----------
        thetas: Iterable[float]
            Flattened list of variational parameters (length = num_qubits * depth).

        Returns
        -------
        expectation: np.ndarray
            Expectation values of the Z observables for each qubit.
        """
        param_dict = {self.weights[i]: thetas[i] for i in range(len(thetas))}
        bound_qc = self.circuit.bind_parameters(param_dict)

        job = execute(bound_qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_qc)

        # Convert counts to expectation values
        expectations = np.zeros(self.num_qubits)
        for bitstring, freq in counts.items():
            prob = freq / self.shots
            bits = np.array([int(b) for b in bitstring[::-1]])  # little‑endian
            z_vals = 1 - 2 * bits  # +1 for 0, -1 for 1
            expectations += prob * z_vals
        return expectations

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Identity encoding; matches the classical interface."""
        return data

    def get_circuit(self) -> QuantumCircuit:
        """Return the raw circuit for inspection or custom execution."""
        return self.circuit


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[HybridClassifier, List[int], List[int], List[SparsePauliOp]]:
    """
    Instantiate the quantum classifier and expose metadata similar to the
    classical build function.

    Returns
    -------
    model: HybridClassifier
        The instantiated quantum circuit object.
    encoding: List[int]
        Indices of input features that are passed through (identity).
    weight_sizes: List[int]
        Length of the variational parameter vector.
    observables: List[SparsePauliOp]
        Pauli‑Z observables for each qubit.
    """
    model = HybridClassifier(num_qubits, depth)
    encoding = list(range(num_qubits))
    weight_sizes = [len(model.weights)]
    observables = model.observables
    return model, encoding, weight_sizes, observables
