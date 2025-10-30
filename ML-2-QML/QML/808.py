"""Quantum fully‑connected layer using Pennylane variational circuits."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp

def FCL(n_qubits: int = 1, layers: int = 2, dev_name: str = "default.qubit") -> qml.QNode:
    """
    Return a variational quantum circuit that mimics a fully‑connected layer.
    The circuit is parameterised by a 1‑D array of angles and outputs the
    expectation value of the Pauli‑Z operator on the first qubit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    layers : int
        Depth of the ansatz.
    dev_name : str
        Pennylane device name (e.g. 'default.qubit', 'qiskit.aer', etc.).

    Returns
    -------
    qml.QNode
        A callable quantum node with a ``run`` method and a ``train`` helper.
    """
    dev = qml.device(dev_name, wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(params: qnp.ndarray) -> qnp.ndarray:
        """Variational circuit returning the expectation of Z on qubit 0."""
        for layer in range(layers):
            for i in range(n_qubits):
                qml.RY(params[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    class QuantumCircuit:
        def __init__(self):
            self.params = qnp.random.uniform(0, 2 * np.pi, size=(layers, n_qubits))
            self.optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """
            Evaluate the circuit on a single parameter vector.
            The vector is reshaped to match the ansatz dimensions.

            Returns
            -------
            np.ndarray
                The expectation value as a 1‑D array.
            """
            params = qnp.array(thetas).reshape((layers, n_qubits))
            expval = circuit(params)
            return np.array([expval])

        def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            verbose: bool = False,
        ) -> None:
            """
            Train the variational circuit using mean‑squared error loss.

            Parameters
            ----------
            X : np.ndarray
                Input features of shape (n_samples, n_qubits * layers).
            y : np.ndarray
                Target values of shape (n_samples,).
            epochs : int
                Number of training iterations.
            verbose : bool
                If True, print loss every 20 epochs.
            """
            for epoch in range(1, epochs + 1):
                loss = 0.0
                for x, target in zip(X, y):
                    params = qnp.array(x).reshape((layers, n_qubits))
                    expval = circuit(params)
                    loss += (expval - target) ** 2
                    grads = qml.grad(circuit)(params)
                    self.params = self.optimizer.step(grads, self.params)
                loss /= len(X)
                if verbose and epoch % 20 == 0:
                    print(f"Epoch {epoch:3d} – Loss: {loss:.6f}")

    return QuantumCircuit()
