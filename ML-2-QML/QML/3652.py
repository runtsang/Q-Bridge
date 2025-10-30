from __future__ import annotations

import torch
import torchquantum as tq
import numpy as np

class HybridSelfAttention(tq.QuantumModule):
    """
    Quantum‑enhanced self‑attention that mirrors the classical version.
    The circuit encodes each 2×2 patch into a 4‑qubit register, applies a
    random variational layer, and measures all qubits.  The measurement
    outcomes are treated as a feature map that is fed into a classical
    scaled‑dot‑product attention.
    """

    def __init__(self, n_qubits: int = 4, embed_dim: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        # Encoder that maps 4 classical inputs into qubit rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            Tensor of shape (B, 1, H, W) containing grayscale images.
        rotation_params, entangle_params
            Optional quantum hyper‑parameters. They are ignored here but
            can be used to condition the variational layer if desired.
        Returns
        -------
        output
            Tensor of shape (B, embed_dim) after attention aggregation.
        """
        B = inputs.shape[0]
        device = inputs.device
        # Prepare quantum device
        qdev = tq.QuantumDevice(self.n_qubits, bsz=B, device=device)
        # Extract 2×2 patches
        patches = []
        H, W = inputs.shape[2], inputs.shape[3]
        for r in range(0, H, self.kernel_size):
            for c in range(0, W, self.kernel_size):
                # Flatten 2×2 patch into 4 values
                patch = torch.stack(
                    [
                        inputs[:, 0, r, c],
                        inputs[:, 0, r, c + 1],
                        inputs[:, 0, r + 1, c],
                        inputs[:, 0, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode into qubits
                self.encoder(qdev, patch)
                # Variational layer
                self.q_layer(qdev)
                # Measure all qubits
                meas = self.measure(qdev)
                patches.append(meas.view(B, self.n_qubits))
        # Concatenate patch features
        feat = torch.cat(patches, dim=1)  # (B, N * n_qubits)
        N = feat.shape[1] // self.n_qubits
        feat = feat.view(B, N, self.n_qubits)  # (B, N, Q)
        # Classical attention over patches
        scores = torch.softmax((feat @ feat.transpose(1, 2)) / np.sqrt(self.n_qubits), dim=-1)
        out = torch.bmm(scores, feat)  # (B, N, Q)
        # Aggregate over patches
        return out.mean(dim=1)  # (B, Q)

__all__ = ["HybridSelfAttention"]
