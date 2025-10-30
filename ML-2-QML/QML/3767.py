import torch
import torchquantum as tq
import torch.nn.functional as F

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum two‑qubit kernel applied to 2×2 image patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        Tensor
            Flattened quantum features of shape (B, 4*14*14).
        """
        B = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=B, device=x.device)
        x = x.view(B, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(B, 4))
        return torch.cat(patches, dim=1)  # (B, 4*14*14)

class SelfAttention(tq.QuantumModule):
    """Hybrid quantum self‑attention that first extracts quanvolution features and then applies a
    variational attention circuit. The attention weights are derived from measurement
    probabilities and used to weight the classical feature vector."""
    def __init__(self, n_qubits: int = 4, embed_dim: int = 4):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        # Variational attention circuit
        self.attn_encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.attn_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        Tensor
            Quantum‑attention‑weighted features of shape (B, embed_dim).
        """
        # 1. Classical‑style quanvolution to obtain a feature vector
        features = self.qfilter(x).to(x.device)  # (B, 4*14*14)
        # 2. Reduce to `n_qubits` values that will drive the attention circuit
        #    We take the mean of every block of 16 features to obtain one value per qubit.
        block_size = features.shape[1] // self.n_qubits
        attn_inputs = features[:, :self.n_qubits * block_size]
        attn_inputs = attn_inputs.view(-1, self.n_qubits, block_size).mean(-1)  # (B, n_qubits)
        # 3. Run the attention circuit for each batch element
        B = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=B, device=x.device)
        self.attn_encoder(qdev, attn_inputs)
        self.attn_layer(qdev)
        probs = self.measure(qdev).float()  # (B, 2**n_qubits)
        # 4. Convert probabilities to a soft‑max over the `n_qubits` dimensions
        #    We simply use the first `n_qubits` probabilities as a coarse weight vector.
        weights = probs[:, :self.n_qubits]
        weights = F.softmax(weights, dim=-1)  # (B, n_qubits)
        # 5. Weight the original feature vector
        #    Broadcast weights to the full feature dimensionality
        weighted = features * weights.unsqueeze(-1)
        # 6. Reduce to the requested embedding dimension
        out = weighted.sum(-1)[:, None].repeat(1, self.embed_dim)  # (B, embed_dim)
        return out

__all__ = ["SelfAttention", "QuanvolutionFilter"]
