"""Quantum convolution (quanvolution) with a parameter‑shared variational ansatz."""
import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute

class Conv:
    """Drop‑in quantum filter that mirrors the classical :class:`Conv` API.

    The original seed defined a single‑parameter circuit per qubit.  This
    extension replaces the circuit with a *parameter‑shared* ansatz
    that can be trained end‑to‑end with gradient‑based optimizers.
    The circuit contains:
    * a data‑encoding layer (RX) where the angle is set to π if the
      input pixel exceeds ``threshold``; otherwise 0;
    * a shared RZ rotation with a trainable parameter ``theta_shared``
      applied to every qubit;
    * a chain of CNOT gates that entangles all qubits;
    * a measurement of all qubits.
    The ``run`` method accepts a 2‑D array and returns the probability
    of “one” in each qubit after the bind‑parameter set.
    """

    def __init__(self,
                 kernel_size: int,
                 backend: qiskit.providers.backend.BaseBackend,
                 shots: int = 100,
                 threshold: float = 0.0,
                 seed: int | None = None) -> None:
        """Create a quantum filter with trainable parameters."""
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.backend = backend
        self.shots = shots

        # Parameters for data encoding: one per qubit
        self.theta_data = [Parameter(f"theta_data_{i}") for i in range(self.n_qubits)]

        # Shared trainable parameter
        self.theta_shared = Parameter("theta_shared")

        # Build the circuit
        self._circuit = QuantumCircuit(self.n_qubits)

        # Data‑encoding layer
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta_data[i], i)

        self._circuit.barrier()

        # Shared rotation layer
        for i in range(self.n_qubits):
            self._circuit.rz(self.theta_shared, i)

        # Entangling layer (chain of CNOTs)
        for i in range(self.n_qubits - 1):
            self._circuit.cx(i, i + 1)

        self._circuit.barrier()
        self._circuit.measure_all()

        # Pre‑compile once
        self._compiled_circuit = self._circuit.copy()
        if seed is not None:
            np.random.seed(seed)

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))

        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta_data[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(self._compiled_circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._compiled_circuit)

        # Compute average probability of |1> across all qubits
        total_ones = 0
        total_shots = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total_ones += ones * val
            total_shots += val

        return total_ones / (total_shots * self.n_qubits)

    def extra_repr(self) -> str:
        return f"kernel_size={int(self.n_qubits**0.5)}, threshold={self.threshold}, shots={self.shots}"
