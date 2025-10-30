import numpy as np
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QuantumSamplerQNN
from qiskit.primitives import StatevectorSampler


class QuanvCircuit:
    """
    Quantum filter that mimics a classical convolution by encoding
    pixel intensities into rotation angles and applying a random
    entangling circuit.
    """

    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size**2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)

        # Parameterised rotations – one per qubit
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)

        self.circuit.barrier()
        # Add random entanglement
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the filter on a single 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            Shape (kernel_size, kernel_size)

        Returns
        -------
        np.ndarray
            Shape (n_qubits,) – probability of measuring |1> for each qubit.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for d in data:
            bind = {self.theta[i]: (np.pi if val > self.threshold else 0)
                    for i, val in enumerate(d)}
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        counts = job.result().get_counts(self.circuit)

        probs = np.zeros(self.n_qubits)
        for key, val in counts.items():
            for i, bit in enumerate(key):
                probs[i] += int(bit) * val
        probs /= (self.shots * self.n_qubits)
        return probs


def SamplerQNN():
    """
    Quantum sampler that first applies a quanvolution filter (QuanvCircuit)
    to the input patch, then feeds the resulting probabilities into a
    parameterised sampler network (QuantumSamplerQNN).
    """
    backend = qiskit.Aer.get_backend("qasm_simulator")
    quanv = QuanvCircuit(kernel_size=2, backend=backend, shots=100, threshold=127)

    # Input parameters are the two averaged probabilities from the filter
    input_params = ParameterVector("x", 2)
    weight_params = ParameterVector("w", 4)

    # The sampler QNN is the same as in the original QNN seed
    sampler = StatevectorSampler()
    qnn = QuantumSamplerQNN(
        circuit=quanv.circuit,
        input_params=input_params,
        weight_params=weight_params,
        sampler=sampler,
    )
    return qnn


__all__ = ["SamplerQNN"]
