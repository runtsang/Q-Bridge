"""Hybrid quantum convolutional classifier.

This module implements a quantum counterpart to the classical
`HybridConvClassifier`.  It uses a parameterised quantum circuit
for the convolutional filter (the “quanvolution” layer) and a shallow
ansatz for the classifier.  The API mirrors the classical version
so that the two can be swapped in an experiment pipeline.

Key features
------------
- The filter circuit is built from a random two‑qubit circuit
  supplemented with RX rotations that encode the pixel values.
- The classifier circuit is a depth‑controlled variational ansatz
  with explicit encoding and CZ entangling gates.
- The `run` method accepts a 2‑D array and returns the probability
  of measuring the all‑`|1>` state after the filter and the mean
  expectation value of the Z observables for the classifier.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational parameters.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


class HybridConvClassifier:
    """
    Quantum analogue of HybridConvClassifier.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel used in the quanvolution filter.
    threshold : float
        Threshold for encoding pixel values into rotation angles.
    num_qubits : int
        Number of qubits used in the classifier circuit.
    depth : int
        Depth of the classifier ansatz.
    shots : int
        Number of shots for circuit execution.
    backend : qiskit.providers.BaseBackend
        Qiskit backend used for execution.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        num_qubits: int = 4,
        depth: int = 2,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Quanvolution filter
        self.filter_circuit = self._build_quanv_circuit()

        # Classifier circuit
        self.classifier_circuit, self.enc_params, self.var_params, self.observables = build_classifier_circuit(num_qubits, depth)

    def _build_quanv_circuit(self) -> QuantumCircuit:
        n_qubits = self.kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(theta[i], i)
        qc.barrier()
        qc += qiskit.circuit.random.random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc

    def _encode_filter(self, data: np.ndarray) -> List[dict]:
        """Return a list of parameter bindings for a single data sample."""
        flat = data.reshape(1, self.kernel_size ** 2)
        binds = []
        for row in flat:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.filter_circuit.parameters, row)}
            binds.append(bind)
        return binds

    def run(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Execute the filter and classifier circuits on the provided data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        Tuple[float, float]
            (probability of measuring |1> across all qubits,
             mean expectation value of the classifier Z observables).
        """
        # Run filter
        filter_binds = self._encode_filter(data)
        job = execute(self.filter_circuit, self.backend, shots=self.shots, parameter_binds=filter_binds)
        result = job.result()
        counts = result.get_counts(self.filter_circuit)

        # Compute average probability of measuring |1> across all qubits
        prob_one = 0.0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            prob_one += ones * val
        prob_one /= self.shots * self.filter_circuit.num_qubits

        # Encode data into classifier circuit
        enc_bind = {p: np.pi if val > self.threshold else 0 for p, val in zip(self.enc_params, data.flatten())}

        # Bind variational parameters to a fixed random seed for reproducibility
        var_bind = {p: 0.0 for p in self.var_params}

        # Run classifier
        job_cls = execute(
            self.classifier_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{**enc_bind, **var_bind}],
        )
        result_cls = job_cls.result()
        counts_cls = result_cls.get_counts(self.classifier_circuit)

        # Compute mean expectation of Z observables
        exp_vals = []
        for obs in self.observables:
            exp = 0.0
            for key, val in counts_cls.items():
                # Map bitstring to Pauli expectation
                parity = 1
                for i, bit in enumerate(reversed(key)):
                    if obs.primitive[i] == "Z" and bit == "1":
                        parity *= -1
                exp += parity * val
            exp /= self.shots
            exp_vals.append(exp)

        return prob_one, np.mean(exp_vals)

    def predict(self, data: np.ndarray) -> float:
        """
        Return the probability of the positive class based on the classifier output.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Probability of class 1.
        """
        _, classifier_score = self.run(data)
        return 1 / (1 + np.exp(-classifier_score))  # logistic mapping


__all__ = ["HybridConvClassifier", "build_classifier_circuit"]
