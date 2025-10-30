import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionAdvanced(nn.Module):
    """
    Hybrid quantum‑classical quanvolution that replaces the classical
    convolution with a parameterised quantum kernel.  Each 2×2 image
    patch is encoded into a 4‑qubit circuit, measured to produce a
    4‑dimensional feature vector, and the resulting sequence is passed
    through a classical self‑attention module before classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        attention_heads: int = 2,
        dropout: float = 0.1,
        device: str = "default.qubit",
        shots: int = 100,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # Quantum kernel parameters
        self.params = nn.Parameter(torch.randn(8))
        self.dev = qml.device(device, wires=4, shots=shots)
        self.qnode = qml.QNode(self._q_circuit, self.dev, interface="torch")
        # Classical attention and classifier
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=attention_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(out_channels * 14 * 14, 10)

    def _q_circuit(self, patch: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Encode the 2×2 patch into Ry rotations
        qml.Ry(patch[0], wires=0)
        qml.Ry(patch[1], wires=1)
        qml.Ry(patch[2], wires=2)
        qml.Ry(patch[3], wires=3)
        # Parameterised rotations
        qml.Ry(params[0], wires=0)
        qml.Ry(params[1], wires=1)
        qml.Ry(params[2], wires=2)
        qml.Ry(params[3], wires=3)
        # Entanglement
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
        # Additional rotations
        qml.Ry(params[4], wires=0)
        qml.Ry(params[5], wires=1)
        qml.Ry(params[6], wires=2)
        qml.Ry(params[7], wires=3)
        # Return expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 10).
        """
        B = x.size(0)
        # Prepare patches
        patches = []
        x_ = x.view(B, 28, 28)
        for r in range(0, 28, self.stride):
            for c in range(0, 28, self.stride):
                patch = torch.stack(
                    [x_[:, r, c], x_[:, r, c + 1], x_[:, r + 1, c], x_[:, r + 1, c + 1]],
                    dim=1,
                )  # (B, 4)
                # Apply quantum circuit to each batch element
                q_out = self.qnode(patch, self.params)  # (B, 4)
                patches.append(q_out)
        # Concatenate patch embeddings: (B, N, 4)
        seq = torch.stack(patches, dim=1)  # (B, N, 4)
        # Prepare for attention: (N, B, C)
        seq = seq.permute(1, 0, 2)  # (N, B, 4)
        attn_out, _ = self.attention(seq, seq, seq)
        attn_out = attn_out.permute(1, 2, 0)  # (B, 4, N)
        attn_out = self.norm(attn_out)
        attn_out = self.dropout(attn_out)
        # Flatten for classification
        flat = attn_out.contiguous().view(B, -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionAdvanced"]
