"""Quantum self‑attention using Pennylane variational circuits.

The circuit implements a parameter‑shaped attention mask over a sequence of
classical embeddings.  The mask is produced by a depth‑2 ansatz that depends
on the input embeddings and a set of trainable rotation angles.  The output
is a probability distribution over the sequence positions that can be used as
attention weights in a hybrid model.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SelfAttention:
    """Variational quantum self‑attention block."""

    def __init__(
        self,
        seq_len: int,
        n_qubits: int = None,
        depth: int = 2,
        device_name: str = "default.qubit",
    ):
        """
        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        n_qubits : int, optional
            Number of qubits; defaults to ``seq_len``.
        depth : int
            Depth of the variational ansatz.
        device_name : str
            Pennylane device to use.
        """
        self.seq_len = seq_len
        self.n_qubits = n_qubits or seq_len
        self.depth = depth
        self.dev = qml.device(device_name, wires=self.n_qubits)

        # Trainable parameters: one rotation per qubit per depth layer
        self.params = pnp.random.uniform(0, 2 * np.pi, (self.depth, self.n_qubits, 3))

        # QNode for forward pass
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, params):
            # Encode classical inputs into rotation angles
            for i, val in enumerate(inputs):
                qml.RX(val, wires=i)
            # Variational layers
            for d in range(self.depth):
                for i in range(self.n_qubits):
                    qml.RY(params[d, i, 0], wires=i)
                    qml.RZ(params[d, i, 1], wires=i)
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement: probability of each qubit being |1>
            probs = [qml.probs(wires=i)[1] for i in range(self.n_qubits)]
            return probs

        self.circuit = circuit

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute attention weights for a single sequence.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (seq_len,) with real values (e.g., embeddings).

        Returns
        -------
        np.ndarray
            Attention probability distribution of shape (seq_len,).
        """
        probs = self.circuit(inputs, self.params)
        # Normalize to sum to 1
        probs = np.array(probs, dtype=np.float64)
        return probs / probs.sum()

    def train(
        self,
        data_loader,
        epochs: int = 10,
        lr: float = 0.01,
        loss_fn=None,
    ):
        """
        Simple training loop using Pennylane's autograd.

        Parameters
        ----------
        data_loader : iterable
            Yields tuples (inputs, target_weights) where ``inputs`` is
            (seq_len,) and ``target_weights`` is a probability vector.
        epochs : int
            Number of epochs.
        lr : float
            Learning rate for the Adam optimizer.
        loss_fn : callable, optional
            Loss function taking (pred, target). Defaults to mean squared error.
        """
        if loss_fn is None:
            def loss_fn(pred, target):
                return np.mean((pred - target) ** 2)

        opt = qml.AdamOptimizer(stepsize=lr)

        for epoch in range(epochs):
            for inputs, target in data_loader:
                def cost_fn(params):
                    return loss_fn(self.circuit(inputs, params), target)

                self.params = opt.step(cost_fn, self.params)
