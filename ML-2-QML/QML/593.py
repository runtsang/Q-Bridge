import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Variational quantum self‑attention block.
    Implements a parameterised circuit with single‑qubit rotations and CNOT entanglement.
    The circuit is differentiable via the parameter‑shift rule, enabling hybrid training.
    """
    def __init__(self, num_qubits: int = 4, wires: list[int] | None = None):
        self.num_qubits = num_qubits
        self.wires = wires if wires is not None else list(range(num_qubits))
        self.dev = qml.device('default.qubit', wires=self.wires)

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def circuit(params, entangle_params):
            # params shape: (num_qubits, 3)
            # entangle_params shape: (num_qubits-1,)
            for i in range(self.num_qubits):
                qml.RX(params[i, 0], wires=self.wires[i])
                qml.RY(params[i, 1], wires=self.wires[i])
                qml.RZ(params[i, 2], wires=self.wires[i])
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[self.wires[i], self.wires[i+1]])
                qml.RX(entangle_params[i], wires=self.wires[i+1])
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def run(self, params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return expectation values of PauliZ on each qubit.

        Parameters
        ----------
        params : np.ndarray of shape (num_qubits, 3)
            Rotation angles for each qubit.
        entangle_params : np.ndarray of shape (num_qubits-1,)
            Entanglement rotation angles for each CNOT+RX pair.

        Returns
        -------
        results : np.ndarray of shape (num_qubits,)
            Expectation values of PauliZ.
        """
        return self.circuit(params, entangle_params).detach().numpy()
