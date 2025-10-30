"""Quantum convolutional filter (quanvolution) using Qiskit."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class Conv:
    """
    Quantum filter that emulates a convolutional kernel.
    Uses a parameterized RX rotation per qubit followed by a simple entangling layer.
    The circuit is reused for each kernel position.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.seed = seed

        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

        # Build the parameterized circuit
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)

        # Simple entangling layer (chain of CNOTs)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)

        self.circuit.barrier()
        self.circuit.measure_all()

        # Transpile once for efficiency
        self.transpiled = transpile(self.circuit, self.backend)

    def run(self, data) -> float:
        """
        Run the quantum circuit on classical data.
        Data is a 2D array of shape (kernel_size, kernel_size).
        Returns the average probability of measuring |1> across all qubits.
        """
        data = np.asarray(data).reshape(1, self.n_qubits)

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.transpiled,
            self.backend,
            parameter_binds=param_binds,
            shots=self.shots,
            seed_simulator=self.seed,
        )
        result = job.result()
        counts = result.get_counts()

        total_ones = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * cnt

        avg_prob = total_ones / (self.shots * self.n_qubits)
        return avg_prob

__all__ = ["Conv"]
