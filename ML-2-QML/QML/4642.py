import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class HybridConv:
    """
    Quantum hybrid convolution that encodes a 2‑D image patch into a Qiskit circuit,
    applies a random entangling layer and a parameterised sampler block, and
    averages the probability of measuring |1> across all qubits.
    
    Parameters
    ----------
    kernel_size : int
        Size of the image patch (kernel). Determines the number of qubits.
    threshold : float
        Value used to binarise input data before encoding.
    shots : int
        Number of shots for the qasm simulator.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.5,
                 shots: int = 512):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size * kernel_size
        self.backend = Aer.get_backend('qasm_simulator')

        # Input encoding (Ry rotations)
        self.input_params = ParameterVector('x', self.n_qubits)

        # Entangling layer (random circuit)
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.ry(self.input_params[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)

        # Sampler block (parameterised rotations + CX)
        self.weight_params = ParameterVector('w', 4)
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)
        self.circuit.cx(0, 1)

        # Measurement
        self.circuit.measure_all()

        # SamplerQNN wrapper
        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler
        )

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum convolution on a 2‑D numpy array.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Flatten and binarise input according to threshold
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {p: np.pi if val > self.threshold else 0
                    for p, val in zip(self.input_params, dat)}
            param_binds.append(bind)

        # Run sampler network
        probs = self.sampler_qnn.run(param_binds)

        # Compute expected number of |1> outcomes
        exp_ones = 0.0
        for prob_dict in probs:
            for bitstring, p in prob_dict.items():
                ones = bitstring.count('1')
                exp_ones += ones * p
        exp_ones /= (self.shots * self.n_qubits)
        return exp_ones

__all__ = ["HybridConv"]
