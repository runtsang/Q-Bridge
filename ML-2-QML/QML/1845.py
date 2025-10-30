import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter, QuantumCircuit
from typing import Iterable

class FCLayer:
    """
    Quantum fully connected layer implemented as a variational circuit.

    The circuit consists of Hadamard gates, a layer of Ry rotations
    parameterized by ``thetas``, and measurement of the first qubit
    in the computational basis.  The expectation value of Pauli‑Z
    on qubit 0 is returned as the layer output, mirroring the
    classical tanh activation in the original seed.

    Parameters
    ----------
    n_qubits : int, optional
        Number of qubits in the variational circuit. Defaults to 1.
    backend : qiskit.providers.BaseBackend, optional
        Backend used to execute the circuit. Defaults to the Aer
        qasm simulator.
    shots : int, optional
        Number of shots for the simulation. Defaults to 1024.
    """

    def __init__(self, n_qubits: int = 1,
                 backend: qiskit.providers.BaseBackend = None,
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self._create_circuit()

    def _create_circuit(self) -> None:
        """Build the parameterized variational circuit."""
        self.params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(range(self.n_qubits))
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        self.circuit.barrier()
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit with the supplied parameters and return
        the expectation value of Pauli‑Z on qubit 0.

        Parameters
        ----------
        thetas : iterable of float
            Parameter vector of length ``n_qubits``. Each element
            corresponds to the rotation angle for a Ry gate.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value.
        """
        thetas = list(thetas)
        if len(thetas)!= self.n_qubits:
            raise ValueError("Parameter vector length must equal number of qubits.")
        bind_dict = {p: t for p, t in zip(self.params, thetas)}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[bind_dict])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array([counts.get(bit, 0) for bit in result.get_counts(self.circuit).keys()], dtype=float) / self.shots
        outcomes = np.array([1 if bit[0] == '0' else -1 for bit in result.get_counts(self.circuit).keys()], dtype=float)
        expectation = np.sum(outcomes * probs)
        return np.array([expectation])

    def get_params(self) -> np.ndarray:
        """Return the current parameter vector."""
        return np.array([p.value if p.value is not None else 0.0 for p in self.params], dtype=float)

    def set_params(self, params: Iterable[float]) -> None:
        """Set the parameter vector to new values."""
        params = list(params)
        if len(params)!= self.n_qubits:
            raise ValueError("Parameter vector length mismatch.")
        self.params = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        self.circuit.h(range(self.n_qubits))
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        self.circuit.barrier()
        self.circuit.measure_all()

__all__ = ["FCLayer"]
