import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute

class FCL:
    """
    Variational fully‑connected layer implemented as a
    parameterised quantum circuit.  Each qubit receives
    a dedicated Ry rotation, followed by a linear chain
    of CNOTs to introduce entanglement.  The output is
    the expectation value of the total Z operator.
    """

    def __init__(self, n_qubits: int = 2, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameterised circuit
        self.circuit = QuantumCircuit(n_qubits)
        self.params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(n_qubits)]

        # Create entangling layer
        self.circuit.h(range(n_qubits))
        for i, p in enumerate(self.params):
            self.circuit.ry(p, i)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Bind the parameter vector to the circuit and
        evaluate the expectation value of the total Z
        operator.

        Parameters
        ----------
        thetas : iterable of float
            Length must match ``n_qubits``.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError("Number of thetas must equal number of qubits.")
        param_bind = {p: t for p, t in zip(self.params, thetas)}

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute expectation of the sum of Z operators
        expectation = 0.0
        for state, cnt in counts.items():
            prob = cnt / self.shots
            # Map |0> -> +1, |1> -> -1 for each qubit
            z_sum = sum(1 if bit == "0" else -1 for bit in state[::-1])
            expectation += z_sum * prob
        return np.array([expectation])

__all__ = ["FCL"]
