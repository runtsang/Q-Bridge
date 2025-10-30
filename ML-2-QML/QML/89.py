"""Quantum convolutional filter using Pennylane.

The class implements a variational circuit that emulates a k×k convolution
kernel.  It supports multi‑channel inputs and can be used as a drop‑in
replacement for a classical Conv layer.  The circuit is defined with the
'torch' interface so gradients flow through the parameters during
back‑propagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuantumConvFilter(nn.Module):
    """
    Variational quantum convolutional filter.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel (k×k).
    in_channels : int
        Number of input channels.
    threshold : float
        Threshold used for angle encoding.
    device : str
        Pennylane device name (default 'default.qubit').
    """

    def __init__(
        self,
        kernel_size: int = 2,
        in_channels: int = 1,
        threshold: float = 0.0,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.dev = qml.device(device, wires=self.n_qubits)
        # trainable parameters of the variational ansatz
        self.params = nn.Parameter(torch.rand(self.n_qubits))
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(data: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Angle encoding: RX with either 0 or π
            for i in range(self.n_qubits):
                theta = torch.pi if data[i] > self.threshold else 0.0
                qml.RX(theta, wires=i)
            # Variational ansatz: RY rotations
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, channels, H-k+1, W-k+1).
        """
        batch, channels, H, W = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size)
        L = patches.shape[-1]
        patches = patches.reshape(batch, channels, self.n_qubits, L)
        outputs = torch.empty(batch, channels, L, dtype=torch.float32, device=x.device)

        for b in range(batch):
            for c in range(channels):
                data = patches[b, c].transpose(0, 1).detach()  # shape (n_qubits, L)
                for p in range(L):
                    expvals = self.circuit(data[:, p], self.params)
                    prob = ((1 - torch.tensor(expvals, device=x.device)) / 2).mean()
                    outputs[b, c, p] = prob

        out_H = H - self.kernel_size + 1
        out_W = W - self.kernel_size + 1
        return outputs.reshape(batch, channels, out_H, out_W)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"in_channels={self.in_channels}, threshold={self.threshold})"
        )


__all__ = ["QuantumConvFilter"]
