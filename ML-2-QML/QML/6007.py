"""Hybrid convolution and fully connected quantum layer implementation using Qiskit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer

class HybridConvFC:
    """
    Quantum implementation of a convolution followed by a fully connected layer.
    The convolution is performed on a grid of qubits with data‑dependent RX
    rotations, followed by a shallow entangling layer.  The fully connected
    layer is a single qubit with a parameterised Ry rotation.  After
    measurement, expectation values are classically combined to produce a
    scalar output.

    Scaling paradigm: combination of quantum convolution + quantum FC.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 conv_shots: int = 200,
                 fc_shots: int = 200,
                 threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.conv_shots = conv_shots
        self.fc_shots = fc_shots

        self.backend = Aer.get_backend("qasm_simulator")

        # Convolution circuit
        self.conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta_params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.conv_circuit.rx(self.theta_params[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += qiskit.circuit.random.random_circuit(self.n_qubits, depth=2)
        self.conv_circuit.measure_all()

        # Fully connected circuit (single qubit)
        self.fc_circuit = QuantumCircuit(1)
        self.fc_theta = qiskit.circuit.Parameter("fc_theta")
        self.fc_circuit.h(0)
        self.fc_circuit.ry(self.fc_theta, 0)
        self.fc_circuit.measure_all()

    def _run_conv(self, data: np.ndarray) -> float:
        """
        Execute the convolution circuit on a single kernel‑sized patch.
        Returns average probability of measuring |1> across qubits.
        """
        param_binds = []
        for i, val in enumerate(data.flatten()):
            bind = {self.theta_params[i]: np.pi if val > self.threshold else 0.0}
            param_binds.append(bind)

        job = execute(self.conv_circuit,
                      self.backend,
                      shots=self.conv_shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.conv_circuit)

        total_ones = 0
        for bitstring, count in result.items():
            ones = bitstring.count('1')
            total_ones += ones * count
        return total_ones / (self.conv_shots * self.n_qubits)

    def _run_fc(self, theta: float) -> float:
        """
        Execute the fully connected circuit with a single parameter.
        Returns expectation value of Z measurement.
        """
        bind = {self.fc_theta: theta}
        job = execute(self.fc_circuit,
                      self.backend,
                      shots=self.fc_shots,
                      parameter_binds=[bind])
        result = job.result().get_counts(self.fc_circuit)

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.fc_shots
        expectation = np.sum(states * probabilities)
        return expectation

    def run(self, data: np.ndarray, fc_theta: float) -> float:
        """
        Full forward pass: first quantum convolution, then quantum fully connected.

        Args:
            data (np.ndarray): 2D array of shape (kernel_size, kernel_size)
            fc_theta (float): parameter for the fully connected layer

        Returns:
            float: scalar output
        """
        conv_out = self._run_conv(data)
        fc_out = self._run_fc(fc_theta)
        # Classical post‑processing: weighted product
        return conv_out * fc_out


__all__ = ["HybridConvFC"]
