import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable

class FCL:
    """
    Variational quantum circuit that implements a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    backend : qiskit.providers.Backend, optional
        Qiskit backend to execute the circuit.  Defaults to the Aer qasm simulator.
    shots : int
        Number of shots for state‑vector estimation.
    """

    def __init__(self, n_qubits: int = 1,
                 backend: qiskit.providers.Backend | None = None,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Create parameter list
        self.params = [Parameter(f"θ{i}") for i in range(n_qubits)]

        # Build the circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Entangling layer
        for i in range(n_qubits):
            self.circuit.h(i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Parameterized Ry rotations
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        # Measurement
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return the
        expectation value of the computational basis label.

        Parameters
        ----------
        thetas : iterable of float
            Length must match ``n_qubits``.

        Returns
        -------
        np.ndarray
            Array of shape (1,) containing the expectation value.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {len(thetas)}")

        bound_circuit = self.circuit.bind_parameters(
            {p: v for p, v in zip(self.params, thetas)}
        )
        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        # Convert bitstrings to integer values
        probs = np.array([counts.get(bs, 0) for bs in sorted(counts)]) / self.shots
        states = np.array([int(bs, 2) for bs in sorted(counts)])

        expectation = np.sum(states * probs)
        return np.array([expectation])
