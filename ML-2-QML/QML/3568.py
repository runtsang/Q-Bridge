import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridQuantumSamplerQNN:
    """
    Quantum hybrid sampler network.

    Builds a random quantum convolution circuit followed by a Qiskit
    SamplerQNN.  The convolution output is interpreted as a probability
    that is fed into the sampler QNN, producing a two‑class output.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 shots: int = 200,
                 threshold: float = 127) -> None:
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        # ----- Quantum convolution circuit -----
        self.conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.conv_circuit.rx(self.theta[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(self.n_qubits, 2)
        self.conv_circuit.measure_all()

        # ----- Sampler QNN circuit -----
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        for w in weights:
            qc.ry(w, 0)
            qc.ry(w, 1)
            qc.cx(0, 1)
        sampler = StatevectorSampler(self.backend)
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler
        )

    def run(self, image: np.ndarray) -> np.ndarray:
        """
        Execute the hybrid quantum sampler on a 2×2 image.

        Parameters
        ----------
        image : np.ndarray
            2D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        np.ndarray
            Two‑class probability array returned by the sampler QNN.
        """
        # Bind parameters for the convolution circuit
        bind_map = {self.theta[i]: np.pi if val > self.threshold else 0
                    for i, val in enumerate(image.flatten())}
        job = execute(self.conv_circuit, self.backend,
                      shots=self.shots,
                      parameter_binds=[bind_map])
        counts = job.result().get_counts(self.conv_circuit)

        # Compute average probability of measuring |1> over all qubits
        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += sum(int(b) for b in bitstring) * freq
        prob = total_ones / (self.shots * self.n_qubits)

        # Feed this probability into the sampler QNN
        output = self.sampler_qnn(prob)
        return output

__all__ = ["HybridQuantumSamplerQNN"]
