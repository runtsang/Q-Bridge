"""Hybrid quantum‑classical classifier with self‑attention sub‑circuit.

The quantum implementation follows the same public interface as the
original ``QuantumClassifierModel`` but now embeds a dedicated
self‑attention block that is executed before the variational ansatz.
Both the encoding and the attention parameters are treated as
variational parameters, allowing the whole circuit to be trained
with a hybrid optimiser.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a layered ansatz with data‑encoding and variational layers.

    Parameters
    ----------
    num_qubits:
        Number of qubits in the circuit.
    depth:
        Number of variational layers.

    Returns
    -------
    circuit:
        The full quantum circuit.
    encoding:
        ParameterVector for data‑encoding.
    weights:
        ParameterVector for variational parameters.
    observables:
        Pauli operators used for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data‑encoding layer
    for i, param in enumerate(encoding):
        qc.rx(param, i)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            qc.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

    # Measurement observables
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return qc, encoding, weights, observables


class QuantumSelfAttention:
    """
    Quantum self‑attention sub‑circuit.

    The circuit implements a small block that applies single‑qubit
    rotations followed by controlled‑X gates, mirroring the classical
    self‑attention pattern.  The parameters are treated as variational
    and can be updated during training.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class QuantumClassifierModel:
    """
    Quantum wrapper that mirrors the classical API.

    The model first runs the self‑attention circuit to produce a
    measurement‑based feature vector, encodes these features into
    the main classifier circuit, and finally measures the output
    observables.  The returned probabilities are compatible with
    the classical ``QuantumClassifierModel``.
    """

    def __init__(self, num_qubits: int, depth: int, backend=None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Main classifier circuit
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )

        # Self‑attention block
        self.attention = QuantumSelfAttention(num_qubits)

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the full hybrid circuit.

        Parameters
        ----------
        inputs:
            Raw input data of shape ``(num_qubits,)``.
        rotation_params, entangle_params:
            Parameters for the attention sub‑circuit.
        shots:
            Number of shots for measurement.
        """
        # 1. Run attention to obtain a measurement string
        attn_counts = self.attention.run(self.backend, rotation_params, entangle_params, shots=shots)

        # Convert counts to a probability distribution over basis states
        probs = np.zeros(2 ** self.num_qubits)
        for state, count in attn_counts.items():
            idx = int(state[::-1], 2)  # little‑endian ordering
            probs[idx] = count / shots

        # 2. Use the resulting probabilities as data‑encoding parameters
        #    (here we simply map the first num_qubits probabilities to the encoding)
        encoded = probs[: self.num_qubits]
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters(
            {self.encoding[i]: encoded[i] for i in range(self.num_qubits)}
        )

        # 3. Execute the classifier circuit
        job = execute(bound_circuit, self.backend, shots=shots)
        result = job.result()
        # Expectation values of Z observables
        exp_vals = np.array(
            [result.get_expectation_value(obs, bound_circuit) for obs in self.observables]
        )
        return exp_vals

    def predict(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Return class probabilities.
        """
        exp_vals = self.run(inputs, rotation_params, entangle_params, shots)
        probs = np.exp(exp_vals) / np.sum(np.exp(exp_vals))
        return probs


__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "QuantumSelfAttention"]
