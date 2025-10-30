"""GraphQNNHybrid: Quantum variational circuit that refines a predicted unitary.

The class takes an initial unitary (typically the prediction from the
classical GNN) and trains a parameterised Pennylane circuit to maximise
fidelity with the true target unitary.  The circuit is built from
StronglyEntanglingLayers, and the loss is 1‑fidelity.  The `train`
method runs a simple gradient‑descent loop and `predict` returns the
refined unitary as a NumPy array.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import pennylane.numpy as npq
from typing import Tuple

class GraphQNNHybrid:
    """Hybrid variational circuit that refines a unitary prediction."""

    def __init__(self, num_qubits: int, init_unitary: np.ndarray | None = None,
                 device: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.device = qml.device(device, wires=num_qubits)
        self.init_unitary = init_unitary
        # initialise parameters for StronglyEntanglingLayers
        self.params = npq.random.uniform(0, 2 * np.pi,
                                         (1, num_qubits, 3))
        self.opt = qml.AdamOptimizer(stepsize=0.01)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            if self.init_unitary is not None:
                qml.MatrixProductGate(self.init_unitary, wires=range(self.num_qubits))
            qml.StronglyEntanglingLayers(params, wires=range(self.num_qubits))
            return qml.state()
        return circuit

    def fidelity(self, state_a: np.ndarray, state_b: np.ndarray) -> float:
        """Return the squared overlap between two pure state vectors."""
        return np.abs(np.vdot(state_a, state_b)) ** 2

    def train(self, target_unitary: np.ndarray, steps: int = 200,
              lr: float = 0.01, verbose: bool = False):
        """Train the variational parameters to maximise fidelity with
        ``target_unitary``.  The optimisation uses a simple Adam loop.
        """
        # fixed initial state |0...0>
        init_state = np.zeros((2 ** self.num_qubits,), dtype=complex)
        init_state[0] = 1.0
        # target state = target_unitary * init_state
        target_state = target_unitary @ init_state
        params = self.params
        for step in range(steps):
            loss, grads = self.opt.step_and_cost(
                lambda p: 1.0 - self.fidelity(self._circuit(p), target_state),
                params)
            params = grads
            if verbose and step % 20 == 0:
                print(f"Step {step:4d} loss: {loss:.6f}")
        self.params = params

    def predict(self) -> np.ndarray:
        """Return the unitary matrix represented by the trained circuit."""
        return qml.matrix(self._circuit, self.params).numpy()

__all__ = [
    "GraphQNNHybrid",
]
