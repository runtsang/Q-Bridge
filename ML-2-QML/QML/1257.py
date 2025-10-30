"""Quantum‑classical quanvolutional filter with a learnable variational circuit."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionGen203(tq.QuantumModule):
    """
    Hybrid quanvolutional filter that applies a 2×2 patch encoding,
    a learnable 4‑qubit variational circuit, and a multi‑head attention
    head before a linear classifier.
    """

    def __init__(self, n_wires: int = 4, n_heads: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.n_heads = n_heads

        # Encode classical pixel values into qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Learnable variational circuit (3 layers of Ry and CNOT)
        self.var_circuit = tq.QuantumCircuit(
            [
                {"func": "ry", "wires": [0], "params": ["theta0"]},
                {"func": "ry", "wires": [1], "params": ["theta1"]},
                {"func": "ry", "wires": [2], "params": ["theta2"]},
                {"func": "ry", "wires": [3], "params": ["theta3"]},
                {"func": "cx", "wires": [0, 1]},
                {"func": "cx", "wires": [2, 3]},
                {"func": "ry", "wires": [0], "params": ["theta4"]},
                {"func": "ry", "wires": [1], "params": ["theta5"]},
                {"func": "ry", "wires": [2], "params": ["theta6"]},
                {"func": "ry", "wires": [3], "params": ["theta7"]},
                {"func": "cx", "wires": [0, 1]},
                {"func": "cx", "wires": [2, 3]},
            ]
        )

        # Learnable parameters for the variational circuit
        self.theta = nn.Parameter(torch.randn(8))

        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical attention head
        self.attn = nn.MultiheadAttention(
            embed_dim=4, num_heads=self.n_heads, batch_first=True
        )

        # Linear classifier
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            log‑softmax logits of shape (batch, 10)
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Reshape to (batch, 28, 28)
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode classical data into qubit rotations
                self.encoder(qdev, data)

                # Apply the learnable variational circuit
                params = [self.theta[i] for i in range(8)]
                self.var_circuit.apply(qdev, params)

                # Measure qubits
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))

        # Concatenate all patch measurements
        features = torch.cat(patches, dim=1)  # (batch, 4 * 14 * 14)

        # Apply attention: reshape to (batch, seq_len, embed_dim)
        attn_in = features.view(bsz, 14 * 14, 4)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        attn_out = attn_out.reshape(bsz, -1)

        logits = self.linear(attn_out)
        return F.log_softmax(logits, dim=-1)
