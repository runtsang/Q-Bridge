import numpy as np
import qiskit
from qiskit import execute
from qiskit.circuit import Parameter
from typing import Iterable, Sequence, Union


class HybridFCL:
    """
    Parameterized quantum circuit designed to act as the quantum
    component of a hybrid FCL model. The circuit can be run on
    a simulator or a real device and supports a configurable
    number of shots.  It expects a single parameter per qubit
    and returns the expectation value of the computational basis
    measurement on the first qubit.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        backend (str | qiskit.providers.BaseBackend): Quantum backend.
        shots (int): Number of shots for measurement.
    """
    def __init__(
        self,
        n_qubits: int = 1,
        backend: Union[str, qiskit.providers.BaseBackend] = "qasm_simulator",
        shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.shots = shots
        # Resolve backend
        if isinstance(backend, str):
            self.backend = qiskit.Aer.get_backend(backend)
        else:
            self.backend = backend

        # Parameter for each qubit
        self.thetas = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        # Simple ansatz: H on all qubits, then RX(theta) on each qubit
        self.circuit.h(range(n_qubits))
        for i, theta in enumerate(self.thetas):
            self.circuit.rx(theta, i)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[Sequence[float]]) -> np.ndarray:
        """
        Execute the circuit for a batch of parameter sets.

        Parameters
        ----------
        thetas : Iterable[Sequence[float]]
            Iterable of parameter vectors, one per sample.

        Returns
        -------
        np.ndarray
            Array of expectation values of shape (batch, 1).
        """
        results = []
        for theta_vec in thetas:
            # Bind parameters
            bind_dict = {theta: val for theta, val in zip(self.thetas, theta_vec)}
            bound_circuit = self.circuit.bind_parameters(bind_dict)
            job = execute(bound_circuit, self.backend, shots=self.shots)
            counts = job.result().get_counts(bound_circuit)
            # Compute expectation value of Z on first qubit as example
            expectation = 0.0
            total = 0
            for outcome, cnt in counts.items():
                # outcome string like '01' where leftmost is qubit 0
                z = 1 if outcome[0] == '0' else -1
                expectation += z * cnt
                total += cnt
            expectation /= total
            results.append(expectation)
        return np.array(results).reshape(-1, 1)


__all__ = ["HybridFCL"]
