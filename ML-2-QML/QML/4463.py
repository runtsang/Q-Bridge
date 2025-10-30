import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli

class HybridQuantumCircuit:
    """
    Parameterized quantum circuit that encodes two rotation angles per qubit
    and returns the expectation value of Pauli‑Z on each qubit.
    """
    def __init__(self, num_qubits: int, backend=None, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameters: two per qubit
        self.theta_x = ParameterVector("theta_x", num_qubits)
        self.theta_y = ParameterVector("theta_y", num_qubits)

        # Build circuit
        self.circuit = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            self.circuit.rx(self.theta_x[i], i)
            self.circuit.ry(self.theta_y[i], i)
        # Entanglement
        for i in range(num_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, parameter_sets: list[list[float]]) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameter sets.

        Parameters
        ----------
        parameter_sets : list[list[float]]
            Each inner list contains 2 * num_qubits angles:
            [theta_x0, theta_y0, theta_x1, theta_y1,...].

        Returns
        -------
        np.ndarray
            Array of shape (batch, num_qubits) with expectation values of
            Pauli‑Z on each qubit.
        """
        results = []
        for params in parameter_sets:
            binding = {}
            for i in range(self.num_qubits):
                binding[self.theta_x[i]] = params[2 * i]
                binding[self.theta_y[i]] = params[2 * i + 1]
            bound_circuit = self.circuit.bind_parameters(binding)
            job = execute(bound_circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circuit)
            # Compute expectation of Z for each qubit
            exp_vals = []
            for i in range(self.num_qubits):
                exp = 0.0
                for bitstring, count in counts.items():
                    bit = int(bitstring[::-1][i])
                    z = 1.0 if bit == 0 else -1.0
                    exp += z * count
                exp /= self.shots
                exp_vals.append(exp)
            results.append(exp_vals)
        return np.array(results)

    def sample(self, parameter_sets: list[list[float]]) -> np.ndarray:
        """
        Return raw measurement samples for each parameter set.

        Parameters
        ----------
        parameter_sets : list[list[float]]

        Returns
        -------
        np.ndarray
            Array of shape (batch, shots, num_qubits) with bitstrings.
        """
        samples = []
        for params in parameter_sets:
            binding = {}
            for i in range(self.num_qubits):
                binding[self.theta_x[i]] = params[2 * i]
                binding[self.theta_y[i]] = params[2 * i + 1]
            bound_circuit = self.circuit.bind_parameters(binding)
            job = execute(bound_circuit, self.backend, shots=self.shots)
            result = job.result()
            counts = result.get_counts(bound_circuit)
            bits = []
            for bitstring, count in counts.items():
                bits += [bitstring] * count
            samples.append(np.array(bits))
        return np.array(samples)

__all__ = ["HybridQuantumCircuit"]
