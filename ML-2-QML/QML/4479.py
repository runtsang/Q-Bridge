import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

class HybridFCL:
    """
    Quantum implementation of the hybrid layer. Builds a parameterised
    circuit that emulates a convolutional filter, a fully‑connected block
    and optional fraud‑detection style random layers.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 backend=None,
                 shots: int = 1024,
                 conv_kernel: int = 2,
                 fraud_layers: int = 0):
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots

        # Main circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)

        # Convolutional sub‑circuit
        self.theta_conv = [qiskit.circuit.Parameter(f"theta_c{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.rx(self.theta_conv[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(n_qubits, 2)

        # Fully‑connected block
        self.theta_fc = qiskit.circuit.Parameter("theta_fc")
        self.circuit.barrier()
        self.circuit.ry(self.theta_fc, range(n_qubits))
        self.circuit.barrier()

        # Fraud‑detection style random layers
        for _ in range(fraud_layers):
            self.circuit += random_circuit(n_qubits, 1)

        self.circuit.measure_all()

    def run(self, thetas: list | np.ndarray) -> np.ndarray:
        """
        Execute the circuit with a list of parameters.
        The first ``n_qubits`` values bind to the convolutional parameters,
        the next value to the fully‑connected block, and any remaining
        values are ignored.
        """
        if len(thetas) < len(self.theta_conv) + 1:
            raise ValueError("Insufficient parameters supplied to run the circuit.")
        param_binds = [{p: v for p, v in zip(self.theta_conv, thetas[:len(self.theta_conv)])}]
        param_binds[0][self.theta_fc] = thetas[len(self.theta_conv)]

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()], dtype=float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridFCL"]
