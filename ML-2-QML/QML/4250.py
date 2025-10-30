import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

class EstimatorQNNGen:
    """
    Stand‑alone quantum implementation of the hybrid model.
    Encodes two classical inputs as RX gates, applies a single trainable
    rotation as RY, and measures Z to obtain an expectation value.
    """
    def __init__(self, input_dim=2, weight_dim=1):
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        self.theta = Parameter("theta")
        self.input_params = [Parameter(f"x{i}") for i in range(input_dim)]
        self.circuit = QuantumCircuit(input_dim)
        for i, p in enumerate(self.input_params):
            self.circuit.rx(p, i)
        # Use the last qubit for the weight rotation
        self.circuit.ry(self.theta, input_dim - 1)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, inputs: np.ndarray, weight: float = 0.0) -> np.ndarray:
        """
        Execute the circuit for a batch of inputs.
        :param inputs: (batch, input_dim) array of real numbers
        :param weight: scalar rotation for the RY gate
        :return: (batch,) array of expectation values
        """
        batch = inputs.shape[0]
        expectations = []
        for i in range(batch):
            bind = {p: float(inputs[i, j]) for j, p in enumerate(self.input_params)}
            bind[self.theta] = weight
            bound_qc = self.circuit.bind_parameters(bind)
            job = execute(bound_qc, self.backend, shots=1024)
            counts = job.result().get_counts(bound_qc)
            probs = np.array(list(counts.values())) / 1024
            probs = probs.astype(float)
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            expectation = np.sum(states * probs)
            expectations.append(expectation)
        return np.array(expectations)

__all__ = ["EstimatorQNNGen"]
