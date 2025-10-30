import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.random import random_circuit

class HybridSamplerConv:
    """
    Quantum hybrid model combining a random convolution circuit with a parameterized sampler.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127, backend=None):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Convolution subcircuit
        self.conv_circ = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.conv_circ.rx(self.theta[i], i)
        self.conv_circ.barrier()
        self.conv_circ += random_circuit(self.n_qubits, 2)
        self.conv_circ.measure_all()

        # Sampler subcircuit
        self.input_params = ParameterVector('input', 2)
        self.weight_params = ParameterVector('weight', 4)

        self.sampler_circ = QuantumCircuit(2)
        self.sampler_circ.ry(self.input_params[0], 0)
        self.sampler_circ.ry(self.input_params[1], 1)
        self.sampler_circ.cx(0, 1)
        for w in self.weight_params:
            self.sampler_circ.ry(w, 0)
            self.sampler_circ.ry(w, 1)
        self.sampler_circ.cx(0, 1)
        self.sampler_circ.measure_all()

    def run(self, data, inputs, weights):
        """
        Execute both convolution and sampler circuits.

        Args:
            data: 2D array of shape (kernel_size, kernel_size).
            inputs: array-like of length 2 for sampler input angles.
            weights: array-like of length 4 for sampler weight angles.

        Returns:
            dict with 'conv_prob' and'sampler_prob'.
        """
        # Bind convolution parameters
        conv_bind = {self.theta[i]: np.pi if val > self.threshold else 0
                     for i, val in enumerate(data.flatten())}
        conv_job = execute(self.conv_circ, self.backend, shots=self.shots,
                           parameter_binds=[conv_bind])
        conv_counts = conv_job.result().get_counts(self.conv_circ)

        conv_prob = 0
        for key, val in conv_counts.items():
            ones = sum(int(bit) for bit in key)
            conv_prob += ones * val
        conv_prob /= (self.shots * self.n_qubits)

        # Bind sampler parameters
        sampler_bind = {self.input_params[i]: inputs[i] for i in range(2)}
        sampler_bind.update({self.weight_params[i]: weights[i] for i in range(4)})

        sampler_job = execute(self.sampler_circ, self.backend, shots=self.shots,
                              parameter_binds=[sampler_bind])
        sampler_counts = sampler_job.result().get_counts(self.sampler_circ)

        sampler_prob = 0
        for key, val in sampler_counts.items():
            if key == '11':
                sampler_prob += val
        sampler_prob /= self.shots

        return {"conv_prob": conv_prob, "sampler_prob": sampler_prob}

__all__ = ["HybridSamplerConv"]
