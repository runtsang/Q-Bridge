"""Enhanced classical convolutional filter with optional hybrid fine‑tuning.

The class is intentionally structured so that the same object can be instantiated with either a
`mode='classic'` (default) or `mode='quantum'` backend.  In hybrid mode the **pre‑trained** classical
weights are **transfer‑learned** to a variational circuit that is
optimised with a simple gradient‑based optimiser.  The API mirrors the original
`Conv` function so that downstream code does not need to change.

Typical usage::

    from Conv__gen154 import ConvEnhanced
    conv = ConvEnhanced(kernel_size=3, mode='hybrid')
    out = conv.run(data)

The module demonstrates how a small classical sub‑network can bootstrap a quantum
model, a common pattern in near‑term quantum machine learning.

"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from torch.optim import Adam

__all__ = ["ConvEnhanced"]


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    kernel_size : int
        Size of the convolutional kernel.
    mode : str, optional
        One of ``'classic'``, ``'quantum'`` or ``'hybrid'``.  In
        ``hybrid`` mode the **pre‑trained** classical weights are
        **transfer‑learned** to a variational circuit.
    threshold : float, default=0.0
        Threshold for the conv‑logits.
    learning_rate : float, default=1e-3
        Learning rate for the quantum optimiser.
    num_iterations : int, default=200
        Number of optimisation steps for the quantum part in hybrid mode.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        mode: str = "classic",
        threshold: float = 0.0,
        learning_rate: float = 1e-3,
        num_iterations: int = 200,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # ---------- classic part ----------
        self._classic = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # ---------- quantum part ----------
        if self.mode in ("quantum", "hybrid"):
            self._quantum = self._build_quantum_circuit()
            self._optimizer = Adam(self._quantum.parameters(), lr=self.learning_rate)

        if self.mode == "hybrid":
            # Transfer classical weights into the quantum circuit
            self._transfer_weights()
            # Fine‑tune the quantum circuit
            self._train_quantum(self.num_iterations)

    def _transfer_weights(self) -> None:
        """Copy pretrained classical conv weights into the quantum circuit."""
        # The quantum circuit will use a 2‑D grid of unitary rotations
        # (Rx, Rz).  The weights are mapped to the rotation angles.
        weight = self._classic.weight.detach().cpu().numpy().flatten()
        # Map weight values to [-π, π] range
        weight = 2 * np.pi * (weight - 0.5) / 0.5
        self._quantum.set_parameters(weight)

    # ------------------------------------------------------------------
    #  Quantum circuit building block
    # ------------------------------------------------------------------
    def _build_quantum_circuit(self):
        """Create a random variational circuit that has the same number
        of qubits and *all* the initial parameters.
        """
        import pennylane as qml
        from pennylane import numpy as pnp

        q = self.kernel_size ** 2
        dev = qml.device("default.qubit", wires=q)

        # Initialise a parameter matrix (layers × qubits)
        num_layers = 2
        init_params = pnp.random.uniform(0, 2 * pnp.pi, size=(num_layers, q))

        class QuantumFilter(nn.Module):
            def __init__(self, dev, init_params):
                super().__init__()
                self.dev = dev
                self.params = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))
                @qml.qnode(self.dev, interface="torch")
                def circuit(params, x):
                    # Encode the input
                    for i in range(x.shape[0]):
                        qml.RY(x[i], wires=i)
                    # Variational layers
                    for layer in range(params.shape[0]):
                        for i in range(params.shape[1]):
                            qml.RZ(params[layer, i], wires=i)
                        # Entangling CNOT chain
                        for i in range(q - 1):
                            qml.CNOT(wires=[i, i + 1])
                    # Return expectation of PauliZ on all qubits
                    return qml.expval(qml.PauliZ(0))
                self.circuit = circuit

            def set_parameters(self, flat_params):
                """Map a flat array of angles to the 2‑D parameter matrix."""
                flat_params = np.asarray(flat_params)
                self.params.data = torch.tensor(
                    flat_params.reshape(self.params.shape), dtype=torch.float32
                )

            def forward(self, data):
                # data is a 1‑D torch tensor of shape (q,)
                return self.circuit(self.params, data)

        return QuantumFilter(dev, init_params)

    def _train_quantum(self, iterations: int) -> None:
        """Simple gradient‑based optimisation to maximise the quantum output."""
        for _ in range(iterations):
            self._optimizer.zero_grad()
            # Dummy input: all zeros
            dummy = torch.zeros(self.kernel_size ** 2, dtype=torch.float32)
            out = self._quantum(dummy)
            loss = -out  # maximise output
            loss.backward()
            self._optimizer.step()

    def run(self, data) -> float:
        """Run the filter on a 2‑D array.

        Parameters
        ----------
        data : array‑like
            Input of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Filter response.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        if self.mode == "classic":
            logits = self._classic(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()
        else:  # quantum or hybrid
            # Flatten data and encode into rotation angles
            flat = tensor.view(-1)
            # Encode threshold: values > threshold -> pi, else 0
            flat = torch.where(flat > self.threshold, torch.tensor(np.pi), torch.tensor(0.0))
            out = self._quantum(flat)
            return out.item()
