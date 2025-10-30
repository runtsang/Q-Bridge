"""SelfAttentionV2: Quantum self‑attention block implemented with Pennylane.

The circuit applies a layer of single‑qubit rotations followed by an entangling
CRX gate between neighbouring wires.  The rotation angles are supplied through
``rotation_params`` and the entangling angles through ``entangle_params``.
The module is fully differentiable, enabling gradient‑based training of the
quantum parameters.

The ``run`` method accepts the same signature as the original seed and returns
a NumPy array of expectation values of the Pauli‑Z operator on each wire.
"""

import pennylane as qml
import numpy as np
import torch
from torch import nn


class SelfAttentionV2(nn.Module):
    """Quantum self‑attention block based on a variational circuit."""

    def __init__(self, embed_dim: int = 4, device: str | qml.Device = "default.qubit"):
        super().__init__()
        self.embed_dim = embed_dim
        # Create a Pennylane device
        self.dev = qml.device(device, wires=embed_dim)

        # Parameters will be optimisable tensors
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, 3))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

        # Build the QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, rot: torch.Tensor, ent: torch.Tensor):
            # Encode the classical input as a rotation on each wire
            for i in range(self.embed_dim):
                qml.RX(rot[i, 0], wires=i)
                qml.RY(rot[i, 1], wires=i)
                qml.RZ(rot[i, 2], wires=i)
            # Entangling layer
            for i in range(self.embed_dim - 1):
                qml.CRX(ent[i], wires=[i, i + 1])
            # Expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that evaluates the variational circuit.

        Parameters
        ----------
        inputs : torch.Tensor
            Classical input of shape (batch, embed_dim).  Each element is
            interpreted as a rotation angle for the corresponding qubit.

        Returns
        -------
        torch.Tensor
            Expectation values of shape (batch, embed_dim).
        """
        batch = inputs.shape[0]
        outputs = []
        for i in range(batch):
            out = self.circuit(inputs[i], self.rotation_params, self.entangle_params)
            outputs.append(out)
        return torch.stack(outputs)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Compatibility wrapper that evaluates the circuit on a NumPy array.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles of shape (embed_dim, 3).  Ignored because the
            parameters are trainable; they are passed for API compatibility.
        entangle_params : np.ndarray
            Entangling angles of shape (embed_dim - 1).  Ignored for the same
            reason.
        inputs : np.ndarray
            Classical input of shape (batch, embed_dim).
        shots : int
            Number of shots for the simulator.  Ignored because the default
            Pennylane device is analytic.

        Returns
        -------
        np.ndarray
            Output expectation values of shape (batch, embed_dim).
        """
        self.eval()
        with torch.no_grad():
            inp = torch.tensor(inputs, dtype=torch.float32)
            out = self.forward(inp)
        return out.numpy()

    def train_loop(self, dataloader, loss_fn, optimizer, epochs: int = 10,
                   device: str | torch.device = "cpu") -> None:
        """
        Simple training loop that updates the quantum parameters.

        Parameters
        ----------
        dataloader : Iterable
            Iterable yielding (inputs, targets) batches.
        loss_fn : callable
            Loss function that accepts (outputs, targets).
        optimizer : torch.optim.Optimizer
            Optimizer for the module parameters.
        epochs : int
            Number of epochs to run.
        device : str | torch.device
            Device on which to perform the computation.
        """
        self.to(device)
        for epoch in range(epochs):
            self.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            self.eval()
