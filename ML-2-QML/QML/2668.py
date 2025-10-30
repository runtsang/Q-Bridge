"""Hybrid quantum sampler that combines a quanvolution filter with a parameterized quantum circuit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

class QuanvCircuit:
    """Quantum filter circuit that emulates a convolutional kernel."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
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
        """Return the average probability of measuring |1> across qubits."""
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

class HybridSamplerQNN:
    """Hybrid quantum sampler that first filters data with a quanvolution circuit
    and then feeds the scalar output into a parameterized quantum sampler."""
    def __init__(self, conv_kernel: int = 2, conv_threshold: float = 127, shots: int = 100) -> None:
        self.conv = QuanvCircuit(kernel_size=conv_kernel,
                                 backend=Aer.get_backend("qasm_simulator"),
                                 shots=shots,
                                 threshold=conv_threshold)

        # Define a simple 2‑qubit sampler circuit
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(circuit=qc,
                                       input_params=inputs,
                                       weight_params=weights,
                                       sampler=sampler)

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the hybrid sampler on a 2‑D array."""
        conv_out = self.conv.run(data)
        input_bind = {self.sampler_qnn.input_params[0]: conv_out,
                      self.sampler_qnn.input_params[1]: 0.0}
        return self.sampler_qnn.run(input_bind)

def SamplerQNN() -> HybridSamplerQNN:
    """Return a ready‑to‑use hybrid quantum sampler."""
    return HybridSamplerQNN()

__all__ = ["SamplerQNN"]
