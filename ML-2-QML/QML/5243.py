"""Hybrid quantum‑classical network that mirrors the classical architecture.
The quantum module replaces the classical quanvolution and autoencoder with
parameterised quantum circuits, while the graph‑based adjacency is still
computed classically from the measurement outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1. Quantum quanvolution – 2×2 patches encoded into a 4‑wire circuit
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Apply a 2×2 image patch through a trainable 4‑wire quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encode each pixel with an Ry rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
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
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (B, 784)

# --------------------------------------------------------------------------- #
# 2. Quantum fully‑connected layer – 4‑wire variational circuit
# --------------------------------------------------------------------------- #
class QFCModel(tq.QuantumModule):
    """Four‑wire variational layer that produces a latent quantum state."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Average‑pool the input image to a 4‑dim vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)  # (B, 4)
        return out

# --------------------------------------------------------------------------- #
# 3. Quantum autoencoder – compress 4‑wire state to 3 latent qubits
# --------------------------------------------------------------------------- #
class AutoencoderCircuit(tq.QuantumModule):
    """Variational autoencoder that maps 4 qubits → 3 qubits → 4 qubits."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.ansatz = tq.RandomLayer(n_ops=12, wires=list(range(num_latent + num_trash)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.ansatz(qdev)
        out = self.measure(qdev).view(qdev.bsz, self.num_latent + self.num_trash)
        return out[:, :self.num_latent]

# --------------------------------------------------------------------------- #
# 4. Graph‑based similarity – computed classically from measurement results
# --------------------------------------------------------------------------- #
def compute_adjacency(latents: torch.Tensor, threshold: float = 0.8,
                     secondary: float | None = None, secondary_weight: float = 0.5) -> torch.Tensor:
    cos_sim = F.cosine_similarity(latents.unsqueeze(1), latents.unsqueeze(0), dim=2)
    adj = torch.zeros_like(cos_sim)
    adj[cos_sim >= threshold] = 1.0
    if secondary is not None:
        mask = (cos_sim >= secondary) & (cos_sim < threshold)
        adj[mask] = secondary_weight
    return adj

# --------------------------------------------------------------------------- #
# 5. Hybrid quantum‑classical network
# --------------------------------------------------------------------------- #
class HybridQuanvolutionAutoGraphNet(tq.QuantumModule):
    """Quantum‑classical hybrid that mirrors the classical architecture."""
    def __init__(self, latent_dim: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.compress = nn.Linear(4 * 14 * 14, 4)          # compress classical features
        self.qfc = QFCModel()
        self.autoencoder = AutoencoderCircuit(num_latent=latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantum quanvolution → classical vector
        features = self.quanvolution(x)                      # (B, 784)
        # 2. Classical compression to 4 dims
        compressed = self.compress(features)                 # (B, 4)
        # 3. Encode compressed values into a 4‑wire quantum device
        qdev = tq.QuantumDevice(n_wires=4, bsz=compressed.shape[0], device=x.device)
        for i in range(4):
            qdev.apply(tq.RY(), wires=i, params=compressed[:, i])
        # 4. Variational QFC layer
        self.qfc.q_layer(qdev)
        latent_q = self.qfc.measure(qdev)                    # (B, 4)
        # 5. Autoencoder – compress to latent_dim qubits
        angles = latent_q[:, :self.autoencoder.num_latent]
        qdev_latent = tq.QuantumDevice(n_wires=self.autoencoder.num_latent + self.autoencoder.num_trash,
                                       bsz=compressed.shape[0], device=x.device)
        for i in range(self.autoencoder.num_latent):
            qdev_latent.apply(tq.RY(), wires=i, params=angles[:, i])
        latent_vec = self.autoencoder(qdev_latent)           # (B, latent_dim)
        # 6. Graph refinement (classical)
        adj = compute_adjacency(latent_vec)
        row_sums = adj.sum(dim=1, keepdim=True).clamp_min_(1e-6)
        refined = latent_vec @ (adj / row_sums)
        # 7. Classification
        logits = self.classifier(refined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAutoGraphNet", "QuanvolutionFilter", "QFCModel", "AutoencoderCircuit"]
