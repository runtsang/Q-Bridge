import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridConvSampler:
    """
    Quantum implementation of the hybrid filter‑sampler module.
    It exposes a convolutional filter built from a parameterised quantum circuit
    and a SamplerQNN that produces probability distributions over two outputs.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        conv_threshold: float = 0.5,
        conv_shots: int = 100,
        sampler_shots: int = 200,
    ) -> None:
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv_n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")

        # Build convolution circuit
        theta = ParameterVector("theta", self.conv_n_qubits)
        self.conv_circuit = QuantumCircuit(self.conv_n_qubits)
        for i in range(self.conv_n_qubits):
            self.conv_circuit.rx(theta[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(self.conv_n_qubits, 2)
        self.conv_circuit.measure_all()

        # Sampler QNN circuit
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

        self.sampler = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=StatevectorSampler(),
        )

        self.conv_shots = conv_shots
        self.sampler_shots = sampler_shots

    def run_conv(self, data: np.ndarray) -> float:
        """
        Execute the convolution circuit on a 2‑D array.
        Returns the average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.conv_n_qubits))
        param_binds = []
        for dat in data:
            bind = {
                theta[i]: np.pi if val > self.conv_threshold else 0
                for i, val in enumerate(dat)
            }
            param_binds.append(bind)

        job = execute(
            self.conv_circuit,
            self.backend,
            shots=self.conv_shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.conv_circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.conv_shots * self.conv_n_qubits)

    def run_sampler(self, conv_output: float) -> np.ndarray:
        """
        Use the sampler QNN to produce a probability distribution over two outputs.
        The conv_output is fed as a single scalar input (replicated to match the
        two‑input requirement of the sampler circuit).
        """
        inputs = np.array([[conv_output, conv_output]])
        probs = self.sampler.predict(inputs, self.sampler_shots)[0]
        return probs

    def run(self, data: np.ndarray) -> tuple[float, np.ndarray]:
        conv_out = self.run_conv(data)
        sampler_out = self.run_sampler(conv_out)
        return conv_out, sampler_out

__all__ = ["HybridConvSampler"]
