import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class HybridFullyConnected:
    """
    Parameterised quantum circuit that implements a simple fully connected layer.
    The circuit applies Hadamard gates, parameterised RY rotations, and measures
    the expectation value of the Z operator on the first qubit.
    """

    def __init__(self,
                 n_qubits: int = 1,
                 n_params: int = 1,
                 shots: int = 1024,
                 backend=None):
        self.n_qubits = n_qubits
        self.n_params = n_params
        self.shots = shots
        self.backend = backend or Aer.get_backend('qasm_simulator')

        # Parameter vector
        self.params = ParameterVector('theta', length=n_params)

        # Build circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        for i in range(n_params):
            target = i % n_qubits
            self.circuit.ry(self.params[i], target)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the provided parameters and return the
        expectation value of Z on the first qubit.
        """
        param_bindings = {self.params[i]: float(thetas[i]) for i in range(self.n_params)}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_bindings])
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Convert counts to probabilities
        probs = np.array([counts.get(f'{i:0{self.n_qubits}b}', 0) for i in range(2**self.n_qubits)],
                         dtype=float)
        probs /= self.shots
        # Compute expectation of Z on first qubit
        exp_z = 0.0
        for i, prob in enumerate(probs):
            bin_str = f'{i:0{self.n_qubits}b}'
            z = 1 if bin_str[0] == '0' else -1
            exp_z += z * prob
        return np.array([exp_z])

    def gradient(self, thetas: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Finite‑difference approximation of the gradient of the expectation
        with respect to the parameters.
        """
        grads = np.zeros_like(thetas)
        for i in range(len(thetas)):
            perturbed = thetas.copy()
            perturbed[i] += eps
            f_plus = self.run(perturbed)[0]
            perturbed[i] -= 2 * eps
            f_minus = self.run(perturbed)[0]
            grads[i] = (f_plus - f_minus) / (2 * eps)
        return grads
