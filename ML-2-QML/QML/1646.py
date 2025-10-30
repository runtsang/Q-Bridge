import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

class SelfAttention:
    """
    Variational self‑attention circuit.  Inputs are encoded as RX rotations,
    followed by a trainable rotation on each qubit and a chain of CNOTs
    with optional phase shifts.  The expectation value of Pauli‑Z on each
    qubit is interpreted as an attention score and soft‑maxed to produce
    attention weights over the input sequence.
    """
    def __init__(self, n_qubits: int, wires: list[int] | None = None):
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

        # placeholders for parameters – users will provide them in run()
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(
        self,
        rotation_params: pnp.ndarray,
        entangle_params: pnp.ndarray,
        inputs: pnp.ndarray,
    ):
        # Input encoding: RX proportional to input value
        for i, val in enumerate(inputs):
            qml.RX(val, self.wires[i])

        # Parameterised single‑qubit rotations
        for i in range(self.n_qubits):
            qml.RX(rotation_params[i, 0], self.wires[i])
            qml.RY(rotation_params[i, 1], self.wires[i])
            qml.RZ(rotation_params[i, 2], self.wires[i])

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(self.wires[i], self.wires[i + 1])
            qml.PhaseShift(entangle_params[i], self.wires[i + 1])

        # Measure Pauli‑Z expectation on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in self.wires]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """
        Execute the ansatz and return attention weights.
        Parameters are expected to be NumPy arrays of shapes
        (n_qubits, 3) and (n_qubits - 1,).
        """
        rotation_params = pnp.array(rotation_params, dtype=pnp.float64)
        entangle_params = pnp.array(entangle_params, dtype=pnp.float64)
        inputs = pnp.array(inputs, dtype=pnp.float64)

        # Expectation values
        exps = self.qnode(rotation_params, entangle_params, inputs)

        # Convert to probabilities via softmax
        scores = pnp.exp(exps) / pnp.sum(pnp.exp(exps))
        return scores.astype(np.float64)

__all__ = ["SelfAttention"]
