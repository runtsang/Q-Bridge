import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionEnhanced(tq.QuantumModule):
    """
    Quantum quanvolutional model using a variational circuit per 2x2 patch.
    The circuit contains trainable Ry rotations and a trainable random layer.
    Features are concatenated with a classical skip connection and fed to a
    two‑layer MLP head.
    """
    def __init__(self, patch_size: int = 2, stride: int = 1,
                 hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_wires = 4

        h_out = (28 - patch_size) // stride + 1
        self.skip_linear = nn.Linear(1 * 28 * 28,
                                     4 * h_out * h_out)

        self.mlp = nn.Sequential(
            nn.Linear(8 * h_out * h_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # Trainable variational circuit
        self.q_layer = tq.RandomLayer(n_ops=4, wires=list(range(self.n_wires)),
                                      trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def _quantum_patch(self, qdev: tq.QuantumDevice, data: torch.Tensor):
        """
        Apply the variational circuit to a single 2x2 patch.
        `data` has shape (B, 4) and contains pixel intensities in [0,1].
        """
        # Encode pixel values as Ry rotations
        for i in range(self.n_wires):
            qdev.ry(data[:, i], wires=i)
        # Variational layer
        qdev.apply(self.q_layer)
        return self.measure(qdev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        bsz = x.size(0)
        device = x.device
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, self.stride):
            for c in range(0, 28, self.stride):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1
                )
                # Normalize pixel values to [0, 1]
                patch = patch / 255.0
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                measurement = self._quantum_patch(qdev, patch)
                patches.append(measurement.view(bsz, 4))
        conv_feat = torch.cat(patches, dim=1)  # (B, 4 * H' * W')
        skip_feat = self.skip_linear(x.view(bsz, -1))
        features = torch.cat([conv_feat, skip_feat], dim=1)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)
