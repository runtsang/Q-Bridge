"""Quantum hybrid convolutional module.

The class implements a quantum convolution filter followed by a variational
classifier.  It can be used as a drop‑in replacement for the classical
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit


class HybridConv:
    """Quantum hybrid convolutional block.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolution kernel (defines the number of qubits).
    threshold : float, default=127
        Threshold used to set rotation angles in the convolution circuit.
    shots : int, default=100
        Number of shots for the QASM simulator.
    num_qubits : int, default=1
        Number of qubits in the variational classifier.
    depth : int, default=1
        Depth of the variational ansatz.
    num_classes : int, default=2
        Number of output classes (unused in the 1‑qubit classifier but kept
        for API symmetry).
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 100,
        num_qubits: int = 1,
        depth: int = 1,
        num_classes: int = 2,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # --- Convolution circuit ------------------------------------------
        n_qubits = kernel_size ** 2
        self.conv_circuit = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.conv_circuit.rx(theta[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(n_qubits, 2)
        self.conv_circuit.measure_all()
        self.conv_params = theta

        # --- Variational classifier circuit -------------------------------
        self.classifier = qiskit.QuantumCircuit(num_qubits)
        self.encoding = qiskit.circuit.ParameterVector("x", num_qubits)
        self.weights = qiskit.circuit.ParameterVector("theta", num_qubits * depth)
        for param, qubit in zip(self.encoding, range(num_qubits)):
            self.classifier.rx(param, qubit)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.classifier.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.classifier.cz(qubit, qubit + 1)
        self.classifier.measure_all()
        self.classifier_params = list(self.encoding) + list(self.weights)

    def run(self, data: np.ndarray) -> float:
        """Execute the hybrid block on ``data`` and return a probability.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape ``(kernel_size, kernel_size)`` containing
            classical pixel values.

        Returns
        -------
        float
            Probability of the positive class estimated by the 1‑qubit
            variational classifier.
        """
        # Convolution step -------------------------------------------------
        data_flat = np.reshape(data, (1, self.kernel_size ** 2))
        param_binds = []
        for dat in data_flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.conv_params[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self.conv_circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.conv_circuit)

        # Compute average probability of measuring |1> across all qubits
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        conv_prob = counts / (self.shots * self.kernel_size ** 2)

        # Classifier step -------------------------------------------------
        bind_enc = {self.encoding[i]: conv_prob * np.pi for i in range(self.classifier.num_qubits)}
        job2 = qiskit.execute(
            self.classifier,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind_enc],
        )
        result2 = job2.result().get_counts(self.classifier)

        # Expectation value of Z on the last qubit
        exp_val = 0
        for key, val in result2.items():
            z = 1 - 2 * int(key[-1])  # last qubit
            exp_val += z * val
        exp_val /= self.shots

        prob = (exp_val + 1) / 2
        return prob


__all__ = ["HybridConv"]
