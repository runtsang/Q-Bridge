"""Variational sampler implemented with PennyLane and a hybrid training interface."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer


class EnhancedSamplerQNN:
    """
    A quantum sampler that mirrors the classical EnhancedSamplerQNN.

    The circuit is a 2‑qubit variational ansatz with rotation layers
    followed by controlled‑NOT entanglement.  The QNode returns
    measurement counts (samples) that can be fed into a classical loss
    or used for downstream tasks.

    Parameters
    ----------
    device_name : str, optional
        Backend device name, e.g. ``"default.qubit"`` or a real device like
        ``"ibmq_qasm_simulator"``.  Default is ``"default.qubit"``.
    shots : int, optional
        Number of measurement shots per evaluation.  Default is 1024.
    """

    def __init__(self, device_name: str = "default.qubit", shots: int = 1024) -> None:
        self.device = qml.device(device_name, wires=2, shots=shots)
        self.weight_shape = (4,)  # 4 variational parameters
        self.opt = AdamOptimizer(0.01)

    def circuit(self, inputs: np.ndarray, weights: np.ndarray) -> None:
        """Variational circuit parameterised by ``inputs`` and ``weights``."""
        # Input encoding
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)

        # Entangling layer
        qml.CNOT(0, 1)

        # Parameterised rotation layer
        qml.RY(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(0, 1)
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=1)

    def qnode(self, inputs: np.ndarray) -> qml.QNode:
        """Return a QNode that measures the computational basis."""
        @qml.qnode(self.device, interface="autograd")
        def _qnode(weights: np.ndarray) -> np.ndarray:
            self.circuit(inputs, weights)
            return qml.sample(qml.PauliZ(0))

        return _qnode

    def sample(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a given input and weight vector.

        Returns an array of measurement outcomes (+1/-1) of shape (shots,).

        Parameters
        ----------
        inputs : array_like
            Two real numbers encoded via RY gates.
        weights : array_like
            Four variational parameters.

        Returns
        -------
        np.ndarray
            Measurement samples.
        """
        qnode = self.qnode(inputs)
        return qnode(weights)

    def train_step(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        loss_fn: callable = lambda samp, tgt: np.mean((samp - tgt) ** 2),
    ) -> float:
        """
        Perform one optimisation step on the variational parameters.

        Parameters
        ----------
        inputs : array_like
            Input vector for the circuit.
        targets : array_like
            Target samples (e.g. classical probabilities or labels).
        loss_fn : callable, optional
            A callable that takes the quantum samples and targets and returns a scalar loss.

        Returns
        -------
        float
            The loss value after the update.
        """
        weights = np.random.uniform(0, 2 * np.pi, self.weight_shape)

        def cost(weights):
            samp = self.sample(inputs, weights)
            return loss_fn(samp, targets)

        weights, loss = self.opt.step_and_cost(cost, weights)
        return loss

__all__ = ["EnhancedSamplerQNN"]
