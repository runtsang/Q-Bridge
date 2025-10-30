import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class FCLLayer:
    """
    Variational quantum circuit emulating a fully‑connected layer.
    Supports multiple qubits, entanglement and Z‑expectation on the first qubit.
    """
    def __init__(self, n_qubits: int = 1, backend_name: str = "qasm_simulator", shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend(backend_name)
        self.shots = shots
        self._build_circuit()

    def _build_circuit(self):
        """Construct the ansatz: H on all qubits, Ry(theta_i) on each, CNOT chain, measure."""
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(range(self.n_qubits))
        # Parameterised Ry gates
        self.theta_params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i, theta in enumerate(self.theta_params):
            self.circuit.ry(theta, i)
        # Entanglement: simple linear chain of CNOTs
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the provided parameters.
        Returns the expectation value of Z on qubit 0 as a 1‑D NumPy array.
        """
        param_dict = {param: val for param, val in zip(self.theta_params, thetas)}
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_dict]
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Convert measurement strings to integer states
        probs = np.array([counts.get(state, 0) for state in sorted(counts)]) / self.shots
        states = np.array([int(state, 2) for state in sorted(counts)])
        # Expectation of Z on qubit 0: (-1)**bit0
        bit0 = (states >> (self.n_qubits - 1)) & 1
        expectation = np.sum((1 - 2 * bit0) * probs)
        return np.array([expectation])

    def expectation_z(self, thetas: Iterable[float]) -> float:
        """
        Convenience wrapper that returns a scalar expectation value.
        """
        return float(self.run(thetas)[0])

__all__ = ["FCLLayer"]
