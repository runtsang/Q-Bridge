"""ConvGen224Q: Quantum counterpart to ConvGen224.

The circuit first encodes the 2‑D data into rotation angles, then applies a
randomised convolutional sub‑circuit (mimicking a quanvolution layer).  After
that a layered variational ansatz performs the classification.  The interface
is identical to the classical implementation so that a user can swap between
the two at runtime.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
from typing import List, Tuple


class ConvGen224Q:
    """
    Quantum implementation mirroring the classical ConvGen224.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 100,
        depth: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size:
            Size of the 2‑D kernel; determines the number of qubits.
        threshold:
            Threshold used to decide whether a pixel value triggers a π rotation.
        shots:
            Number of measurement shots per evaluation.
        depth:
            Depth of the variational classifier ansatz.
        """
        self.kernel_size = kernel_size
        self.num_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.depth = depth

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

        # Convolutional sub‑circuit (quanvolution)
        self.conv_circuit = QuantumCircuit(self.num_qubits)
        self.theta = ParameterVector("theta", self.num_qubits)
        for i in range(self.num_qubits):
            self.conv_circuit.rx(self.theta[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(self.num_qubits, depth=2)
        # No measurement here; will be added later

        # Classification ansatz
        self.encoding = ParameterVector("x", self.num_qubits)
        self.weights = ParameterVector("w", self.num_qubits * depth)

        self.class_circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(self.encoding, range(self.num_qubits)):
            self.class_circuit.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(self.num_qubits):
                self.class_circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                self.class_circuit.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        self.observables: List[SparsePauliOp] = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]

        # Combine the two sub‑circuits into one executable circuit
        self.full_circuit = QuantumCircuit(self.num_qubits)
        self.full_circuit.append(self.conv_circuit, range(self.num_qubits))
        self.full_circuit.append(self.class_circuit, range(self.num_qubits))
        self.full_circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the full circuit on the supplied 2‑D data.

        Parameters
        ----------
        data:
            2‑D array of shape (kernel_size, kernel_size) with intensity values.
        Returns
        -------
        float
            Averaged probability of measuring |1> across all qubits,
            interpreted as a probability of the “positive” class.
        """
        # Flatten and normalize data to [0, 1]
        flat = np.reshape(data, (self.num_qubits,))

        # Bind parameters: data -> encoding, threshold -> theta
        param_binds = []
        for val in flat:
            bind = {}
            for i, v in enumerate(flat):
                bind[self.theta[i]] = np.pi if v > self.threshold else 0
                bind[self.encoding[i]] = v
            param_binds.append(bind)

        job = execute(
            self.full_circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.full_circuit)

        # Compute expectation of Z on each qubit
        exp_vals = np.zeros(self.num_qubits)
        total = 0
        for bitstring, freq in counts.items():
            for i, bit in enumerate(bitstring[::-1]):  # reverse due to qiskit ordering
                exp_vals[i] += (1 if bit == "1" else -1) * freq
            total += freq

        exp_vals /= total
        # Average probability of |1> across qubits
        prob_pos = (exp_vals + 1) / 2
        return prob_pos.mean()

    def get_params(self) -> Tuple[List[float], List[float]]:
        """
        Return the current variational parameters (theta, weights).
        Useful for debugging or training with a classical optimiser.
        """
        return [self.theta], [self.weights]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"threshold={self.threshold}, depth={self.depth}, shots={self.shots})"
        )


__all__ = ["ConvGen224Q"]
