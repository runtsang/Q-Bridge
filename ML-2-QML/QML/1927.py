"""Variational quantum circuit for a fully‑connected layer."""

import pennylane as qml
import numpy as np


def FCL() -> object:
    """Return a parameterized quantum circuit.

    The circuit applies Ry rotations to each qubit, entangles them with
    a linear CNOT chain, and measures the expectation value of Pauli‑Z
    on the first qubit.  The interface mirrors the classical counterpart
    with ``run(thetas)`` accepting a list of rotation angles.
    """

    class QuantumCircuit:
        """Parameterized variational circuit."""

        def __init__(self, n_qubits: int = 4, shots: int = 1024) -> None:
            self.n_qubits = n_qubits
            self.device = qml.device("default.qubit", wires=n_qubits, shots=shots)
            self._theta_var = np.random.uniform(0, 2 * np.pi, size=n_qubits)

            @qml.qnode(self.device, interface="numpy")
            def circuit(params: np.ndarray) -> float:
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=i)
                # Linear CNOT chain for entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                return qml.expval(qml.PauliZ(0))

            self._circuit = circuit

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Execute the circuit with the given rotation angles.

            Parameters
            ----------
            thetas : Iterable[float]
                List of Ry rotation angles, one per qubit.

            Returns
            -------
            np.ndarray
                Expected value of Pauli‑Z on the first qubit.
            """
            params = np.array(thetas, dtype=np.float64)
            if params.shape[0]!= self.n_qubits:
                raise ValueError(f"Expected {self.n_qubits} parameters, got {params.shape[0]}")
            expectation = self._circuit(params)
            return np.array([expectation])

    return QuantumCircuit()
