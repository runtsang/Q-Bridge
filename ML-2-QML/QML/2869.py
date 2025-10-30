"""Hybrid quantum model combining a quanvolution filter and a SamplerQNN."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class QuanvCircuit:
    """Quantum convolution filter."""

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, depth=2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Return average probability of measuring |1> across qubits."""
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return counts / (self.shots * self.n_qubits)

class ConvGen133(nn.Module):
    """Quantum version of ConvGen133 combining quanvolution and SamplerQNN."""

    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 100) -> None:
        super().__init__()
        self.conv = QuanvCircuit(kernel_size, shots=shots, threshold=threshold)
        # Build parameterized sampler circuit
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

        sampler = Sampler()
        self.sampler_qnn = QSamplerQNN(circuit=qc,
                                       input_params=inputs,
                                       weight_params=weights,
                                       sampler=sampler)

    def run(self, data: np.ndarray) -> np.ndarray:
        """Return sampler output probabilities for the given image patch."""
        activation = self.conv.run(data)
        input_vals = np.array([activation, activation])
        probs = self.sampler_qnn.predict(input_vals)
        return probs

__all__ = ["ConvGen133"]
