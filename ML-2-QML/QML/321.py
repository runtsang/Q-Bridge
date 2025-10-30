"""Quantum multi‑head self‑attention using Pennylane variational circuits.

The circuit learns attention weights as expectation values of Pauli‑Z on each qubit.
It accepts the same API as the classical module for seamless integration."""
import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

class SelfAttentionGen357:
    """Variational self‑attention implemented with Pennylane."""
    def __init__(self, n_qubits: int = 4, num_heads: int = 2):
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_variational_params()

    def _build_variational_params(self):
        self.rotation_params = pnp.random.uniform(0, 2 * np.pi,
                                                 (self.n_qubits, 3))
        self.entangle_params = pnp.random.uniform(0, 2 * np.pi,
                                                  (self.n_qubits - 1,))

    def _rotation_layer(self, params):
        for i, (rx, ry, rz) in enumerate(params):
            qml.RX(rx, wires=i)
            qml.RY(ry, wires=i)
            qml.RZ(rz, wires=i)

    def _entangle_layer(self, params):
        for i, theta in enumerate(params):
            qml.CRX(theta, wires=[i, i + 1])

    def _attention_circuit(self, rotation_params, entangle_params):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            self._rotation_layer(rotation_params)
            self._entangle_layer(entangle_params)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit()

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Execute the variational attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – RX, RY, RZ angles per qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – CRX angles between adjacent qubits.
        inputs : np.ndarray
            Ignored – kept for API compatibility.

        Returns
        -------
        np.ndarray
            Shape (n_qubits,) – expectation values approximating attention weights.
        """
        rotation_params = rotation_params.reshape(self.n_qubits, 3)
        entangle_params = entangle_params.reshape(self.n_qubits - 1)
        attn_weights = self._attention_circuit(rotation_params, entangle_params)
        return np.array(attn_weights)

    def train_step(self, inputs: np.ndarray, target: np.ndarray, lr: float = 0.01):
        """
        Simple gradient‑descent training of the variational parameters to match a
        target attention vector.

        Parameters
        ----------
        inputs : np.ndarray
            Ignored – placeholder for API consistency.
        target : np.ndarray
            Desired attention weights (shape (n_qubits,)).
        lr : float
            Learning rate.
        """
        def loss_fn(params):
            rot, ent = params
            attn = self._attention_circuit(rot, ent)
            return ((attn - target) ** 2).sum()

        grads = qml.grad(loss_fn)([self.rotation_params, self.entangle_params])
        self.rotation_params -= lr * grads[0]
        self.entangle_params -= lr * grads[1]

__all__ = ["SelfAttentionGen357"]
