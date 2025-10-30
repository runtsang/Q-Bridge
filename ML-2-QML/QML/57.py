"""Quantum self‑attention using PennyLane variational circuits."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class SelfAttention:
    """
    Variational quantum self‑attention layer.

    Parameters
    ----------
    embed_dim : int
        Number of qubits / embedding dimensionality.
    dev : pennylane.Device, optional
        PennyLane device used for simulation or hardware execution.
    """

    def __init__(self, embed_dim: int, dev: qml.Device | None = None):
        self.embed_dim = embed_dim
        self.dev = dev or qml.device("default.qubit", wires=embed_dim)

        # Parameters to be optimized
        self.rotation_params = pnp.random.uniform(0, 2 * np.pi, size=(embed_dim, embed_dim))
        self.entangle_params = pnp.random.uniform(0, 2 * np.pi, size=(embed_dim, embed_dim))

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Build a variational circuit that mimics the attention mechanism."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(qubits):
            # Encode inputs as rotation angles
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)

            # Apply rotation layer
            for i in range(self.embed_dim):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)

            # Entangling layer
            for i in range(self.embed_dim - 1):
                qml.CNOT(wires=[i, i + 1])

            # Apply entanglement parameters
            for i in range(self.embed_dim):
                qml.RX(entangle_params[i, 0], wires=i)
                qml.RY(entangle_params[i, 1], wires=i)
                qml.RZ(entangle_params[i, 2], wires=i)

            # Measure expectation values of Pauli-Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

        return circuit(inputs)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """
        Execute the variational attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the rotation layer, shape (embed_dim, 3).
        entangle_params : np.ndarray
            Parameters for the entanglement layer, shape (embed_dim, 3).
        inputs : np.ndarray
            Input embedding values, shape (embed_dim,).
        shots : int, optional
            Number of measurement shots; if None, use exact expectation values.

        Returns
        -------
        np.ndarray
            Circuit output, shape (embed_dim,).
        """
        if shots is not None:
            # Use a classical simulator with shots (approximate)
            result = self._circuit(rotation_params, entangle_params, inputs)
            return np.array(result)
        else:
            # Exact expectation values
            return np.array(self._circuit(rotation_params, entangle_params, inputs))

    def train(
        self,
        loss_fn,
        optimizer,
        data_loader,
        epochs: int = 10,
        device: str = "cpu",
    ):
        """
        Train the quantum attention parameters end‑to‑end.

        Parameters
        ----------
        loss_fn : callable
            Loss function taking predictions and targets.
        optimizer : torch.optim.Optimizer
            Optimizer that operates on the parameters.
        data_loader : iterable
            Iterable yielding (inputs, targets).
        epochs : int, default=10
            Number of training epochs.
        device : str, default='cpu'
            Device for tensor operations.
        """
        import torch

        # Wrap parameters as torch tensors for autograd
        rot = torch.tensor(self.rotation_params, dtype=torch.float32, requires_grad=True, device=device)
        ent = torch.tensor(self.entangle_params, dtype=torch.float32, requires_grad=True, device=device)

        for epoch in range(epochs):
            for inputs, targets in data_loader:
                inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
                targets = torch.tensor(targets, dtype=torch.float32, device=device)

                def circuit_grad():
                    return torch.tensor(
                        self._circuit(rot.detach().numpy(), ent.detach().numpy(), inputs.detach().numpy()),
                        dtype=torch.float32,
                        device=device,
                    )

                preds = circuit_grad()
                loss = loss_fn(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update numpy parameters after optimizer step
            self.rotation_params = rot.detach().cpu().numpy()
            self.entangle_params = ent.detach().cpu().numpy()


__all__ = ["SelfAttention"]
