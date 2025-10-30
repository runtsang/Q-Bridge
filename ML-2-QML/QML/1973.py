import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class FCLGen:
    """
    Quantum fully‑connected layer implemented with a Pennylane variational
    circuit.  The circuit applies a layer of Ry rotations (parameterized by
    ``thetas``) followed by a configurable number of entanglement layers.
    The expectation value of the Pauli‑Z operator on the first qubit
    represents the layer output.  The ``run`` method accepts a flat
    parameter vector and returns the expectation value as a NumPy array.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        entanglement_layers: int = 2,
        device_name: str | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.entanglement_layers = entanglement_layers
        self.dev = qml.device(device_name or "default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas):
            # Apply parameterised Ry gates
            for i in range(n_qubits):
                qml.RY(thetas[i], wires=i)

            # Entanglement layers
            for _ in range(entanglement_layers):
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Optional long‑range entanglement
                qml.CNOT(wires=[n_qubits - 1, 0])

            # Measure expectation of Z on first qubit
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the variational circuit with the supplied parameters.
        Expects ``thetas`` to have length ``n_qubits``.
        Returns a NumPy array containing the single expectation value.
        """
        theta_arr = pnp.array(list(thetas), dtype=pnp.float64)
        exp_val = self._circuit(theta_arr)
        return np.array([exp_val])

__all__ = ["FCLGen"]
