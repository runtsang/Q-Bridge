"""Hybrid quantum convolution + classifier.

The :class:`HybridConvClassifier` below implements a 2×2 quantum
convolution filter followed by a layered ansatz classifier.  The
filter output is used as an encoding value for the classifier
circuit.  The model is compatible with the classical interface
and can be used in side‑by‑side experiments.

Key ingredients
---------------
* 2×2 quantum filter built with a random circuit and RX rotations.
* Layered variational ansatz (depth configurable) with CZ entanglement.
* Measurement of Z observables to obtain expectation values that
  are interpreted as class scores.
* Uses Qiskit Aer simulator; can be replaced with a real backend.

Usage
-----
>>> from qiskit import Aer
>>> model = HybridConvClassifier(
...     kernel_size=2,
...     depth=2,
...     backend=Aer.get_backend("qasm_simulator"),
...     shots=1024,
...     threshold=0.5,
... )
>>> probs = model.run(np.random.randint(0, 2, (2, 2)))  # shape (4,)
"""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import execute, Aer, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple


def _build_filter_circuit(kernel_size: int) -> Tuple[QuantumCircuit, Iterable, int]:
    """
    Construct a 2×2 quantum filter circuit.

    Parameters
    ----------
    kernel_size : int
        Size of the filter (must be 2 for this implementation).

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised filter circuit.
    theta : list[Parameter]
        List of rotation parameters.
    n_qubits : int
        Number of qubits in the circuit.
    """
    n_qubits = kernel_size ** 2
    circuit = QuantumCircuit(n_qubits)
    theta = ParameterVector("theta", n_qubits)

    # RX rotations controlled by the input amplitude
    for i in range(n_qubits):
        circuit.rx(theta[i], i)

    # Add a small random subcircuit for expressibility
    circuit += random_circuit(n_qubits, depth=2)

    circuit.measure_all()
    return circuit, theta, n_qubits


def _build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered variational classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (equal to the filter output dimensionality).
    depth : int
        Number of alternating Ry–CZ layers.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised classifier circuit.
    encoding : list[Parameter]
        Input encoding parameters.
    weights : list[Parameter]
        Variational parameters.
    observables : list[SparsePauliOp]
        Measurement operators for expectation values.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Initial encoding (RX with data)
    for qubit in range(num_qubits):
        circuit.rx(encoding[qubit], qubit)

    # Alternating layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class HybridConvClassifier:
    """
    Quantum hybrid convolution + classifier.

    Parameters
    ----------
    kernel_size : int, default 2
        Filter size (must be 2 for this implementation).
    depth : int, default 2
        Depth of the classifier ansatz.
    backend : qiskit.providers.base.Provider, optional
        Execution backend; defaults to Aer qasm simulator.
    shots : int, default 1024
        Number of shots per execution.
    threshold : float, default 0.5
        Threshold for the filter rotations.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
    ) -> None:
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")

        self.kernel_size = kernel_size
        self.depth = depth
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        # Build quantum filter
        self.filter_circuit, self.filter_params, self.n_qubits = _build_filter_circuit(
            kernel_size=kernel_size,
        )

        # Build classifier circuit
        self.classifier_circuit, self.encoding, self.weights, self.observables = _build_classifier_circuit(
            num_qubits=self.n_qubits,
            depth=depth,
        )

    def _run_filter(self, data: np.ndarray) -> float:
        """
        Execute the filter circuit on a single 2×2 input.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (2, 2) with values in [0, 1].

        Returns
        -------
        prob : float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.flatten()
        bind = {param: np.pi if val > self.threshold else 0.0 for param, val in zip(self.filter_params, flat)}

        job = execute(
            self.filter_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result().get_counts(self.filter_circuit)
        ones = sum(bitstring.count("1") * val for bitstring, val in result.items())
        return ones / (self.shots * self.n_qubits)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Run the hybrid model on a single input.

        Parameters
        ----------
        data : np.ndarray
            Input image of shape (2, 2) with values in [0, 1].

        Returns
        -------
        probs : np.ndarray
            Class probabilities of shape (num_qubits,).
        """
        # Obtain filter output
        filter_output = self._run_filter(data)

        # Encode filter output into the classifier circuit
        encode_bind = {param: filter_output for param in self.encoding}
        weight_bind = {param: 0.0 for param in self.weights}  # static weights for a demo

        param_bind = {**encode_bind, **weight_bind}

        job = execute(
            self.classifier_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result().get_counts(self.classifier_circuit)

        # Compute expectation values of each observable
        expectations = []
        for i, obs in enumerate(self.observables):
            ones = 0
            for bitstring, val in result.items():
                if bitstring[-1 - i] == "1":
                    ones += val
            exp = (1 - 2 * ones / self.shots)
            expectations.append(exp)

        logits = np.array(expectations)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs


__all__ = ["HybridConvClassifier"]
