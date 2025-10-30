"""HybridSamplerConv: quantum hybrid sampler with quanvolution and QNN sampling."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from qiskit.providers.aer import Aer

class QuanvCircuit:
    """
    Quantum filter circuit used for quanvolution layers.
    """
    def __init__(self, kernel_size: int, backend: qiskit.providers.Backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

def _build_sampler_qnn() -> SamplerQNN:
    """
    Construct a parameterized quantum sampler network.
    """
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)
    qc2 = QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)

    sampler = Sampler()
    return SamplerQNN(circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=sampler)

class HybridSamplerConv:
    """
    Quantum hybrid sampler combining a quanvolution filter and a QNN sampler.
    """
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_shots: int = 100,
                 conv_threshold: float = 127,
                 sampler_shots: int = 1000,
                 backend: qiskit.providers.Backend | None = None) -> None:
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel_size, backend, conv_shots, conv_threshold)
        self.sampler_qnn = _build_sampler_qnn()

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the hybrid quantum sampler on input data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            np.ndarray: sampled probability distribution from the QNN.
        """
        conv_out = self.conv.run(data)
        sampler_input = [conv_out, conv_out]
        return self.sampler_qnn.sample(sampler_input)

__all__ = ["HybridSamplerConv"]
