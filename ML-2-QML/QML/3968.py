"""Hybrid Quantum Classifier - Quantum implementation.

This module implements the same high‑level interface as the classical
version but replaces the convolutional filter with a parameterised
quantum circuit (quanvolution) and the feed‑forward network with a
variational ansatz executed on a Qiskit simulator.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit", "QuanvCircuit"]

def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """Construct a layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the classifier circuit.
    depth : int
        Number of variational layers.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

class QuanvCircuit:
    """Quantum convolutional filter (quanvolution) used as a drop‑in
    replacement for the classical Conv filter.

    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D kernel (assumed square).
    backend : qiskit.providers.BaseBackend
        Backend used to execute the circuit.
    shots : int
        Number of shots for measurement statistics.
    threshold : float
        Threshold used to map classical pixel values to rotation angles.
    """

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [ParameterVector(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on classical data.

        Parameters
        ----------
        data : 2‑D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class HybridQuantumClassifier:
    """Quantum stand‑in for the hybrid classifier.

    The class first applies a quanvolution layer to the input image
    and then runs a variational circuit to compute class logits.  The
    interface mirrors the classical version so that the two can be
    swapped in downstream code.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        conv_kernel: int = 2,
        conv_threshold: float = 127,
        shots: int = 1024,
        backend=None,
    ) -> None:
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel, backend, shots, conv_threshold)
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = backend
        self.shots = shots

    def run(self, data: np.ndarray) -> np.ndarray:
        """Apply quanvolution and variational circuit to obtain logits.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) representing the
            image to classify.
        """
        # 1. Quantum convolution
        conv_output = self.conv.run(data)

        # 2. Encode the conv_output as a single rotation on the first qubit
        param_binds = {self.encoding[0]: conv_output * np.pi}

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        result = job.result().get_counts(self.circuit)

        # 3. Estimate expectation values for each observable
        expectations = []
        for obs in self.observables:
            exp = 0.0
            for state, count in result.items():
                eigenvalue = 1
                for i, qubit in enumerate(obs.paulis):
                    if qubit == "Z" and state[-(i + 1)] == "0":
                        eigenvalue *= 1
                    elif qubit == "Z" and state[-(i + 1)] == "1":
                        eigenvalue *= -1
                exp += eigenvalue * count
            exp /= self.shots
            expectations.append(exp)

        return np.array(expectations)
