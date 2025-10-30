import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from typing import Iterable, List

class FCL:
    """
    Quantum implementation of a fully‑connected layer using a parameterised
    variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode the input vector.
    depth : int
        Number of variational rotation layers.
    backend : qiskit.providers.Backend, optional
        Execution backend.  Defaults to the Aer qasm simulator.
    shots : int
        Number of shots for each evaluation.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        depth: int = 1,
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        self.params = []

        # Encode input via Ry rotations
        for q in range(self.n_qubits):
            theta = Parameter(f"θ_{q}")
            self.circuit.ry(theta, q)
            self.params.append(theta)

        # Variational layers
        for d in range(self.depth):
            for q in range(self.n_qubits):
                phi = Parameter(f"φ_{d}_{q}")
                self.circuit.ry(phi, q)
                self.params.append(phi)
            # Entangling layer (chain of CNOTs)
            for q in range(self.n_qubits - 1):
                self.circuit.cx(q, q + 1)

        # Measurement of all qubits
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a single set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable containing the values for all parameters in the order
            they appear in ``self.params``.
        Returns
        -------
        np.ndarray
            Expected value of Z on the first qubit.
        """
        if len(thetas)!= len(self.params):
            raise ValueError(
                f"Expected {len(self.params)} parameters, got {len(thetas)}"
            )

        bound = {p: val for p, val in zip(self.params, thetas)}
        bound_circuit = self.circuit.bind_parameters(bound)

        job = execute(bound_circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        zero_counts = sum(cnt for bitstring, cnt in counts.items() if bitstring[-1] == '0')
        one_counts  = sum(cnt for bitstring, cnt in counts.items() if bitstring[-1] == '1')
        expectation = (zero_counts - one_counts) / self.shots
        return np.array([expectation])

    def sample(self, thetas: Iterable[float], n_samples: int = 1000) -> np.ndarray:
        """
        Draw samples from the circuit for a given set of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameter values.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing the measured first‑qubit
            outcomes (+1 or -1).
        """
        if len(thetas)!= len(self.params):
            raise ValueError("Parameter count mismatch.")

        bound = {p: val for p, val in zip(self.params, thetas)}
        bound_circuit = self.circuit.bind_parameters(bound)

        job = execute(bound_circuit, self.backend, shots=n_samples)
        result = job.result()
        counts = result.get_counts(bound_circuit)

        outcomes = []
        for bitstring, cnt in counts.items():
            first_bit = int(bitstring[-1])  # qubit 0 is the last bit
            val = 1 if first_bit == 0 else -1
            outcomes.extend([val] * cnt)
        return np.array(outcomes)
