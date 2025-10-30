import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumHybridAttention(nn.Module):
    """
    Quantum‑classical hybrid self‑attention using Pennylane.

    Parameters
    ----------
    embed_dim : int, default=4
        Dimensionality of the input embeddings.
    n_qubits : int, default=4
        Number of qubits in the variational circuit.
    kernel_size : int, default=2
        Size of the convolution kernel applied to the input.
    """
    def __init__(self, embed_dim: int = 4, n_qubits: int = 4, kernel_size: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.kernel_size = kernel_size
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Classical attention layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Convolutional pre‑processor
        self.conv = nn.Conv2d(1, 1, kernel_size, bias=True)

        # Quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_qubits):
            qml.RX(params[3 * i], wires=i)
            qml.RY(params[3 * i + 1], wires=i)
            qml.RZ(params[3 * i + 2], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(wires=range(self.n_qubits)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute hybrid attention output.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        batch, seq, _ = x.shape

        # Convolutional pre‑processor
        x_conv = x.permute(0, 2, 1).unsqueeze(1)
        conv_out = torch.sigmoid(
            self.conv(x_conv[:, :, :self.kernel_size, :self.kernel_size]).mean()
        )

        # Classical attention
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        classical_out = scores @ V

        # Quantum attention
        params = torch.randn(self.n_qubits * 3)
        q_out = self.qnode(params)

        # Combine
        out = classical_out + q_out * conv_out
        return out
