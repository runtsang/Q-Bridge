import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class HybridQuanvolutionRegressionModel(tq.QuantumModule):
    """
    Quantum hybrid model that applies a random two‑qubit quantum kernel to 2x2 image patches
    and then uses linear heads for classification and regression.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4  # one qubit per pixel in a 2x2 patch
        # Encoder that maps pixel intensities to Ry rotations
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
        # Linear heads
        self.classifier = nn.Linear(self.n_wires, num_classes)
        self.regressor = nn.Linear(self.n_wires, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28) or (batch, 28, 28).

        Returns
        -------
        dict
            Dictionary with keys 'logits' and'regression'.
        """
        bsz = x.shape[0]
        device = x.device
        # Ensure shape (batch, 28, 28)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        # Prepare quantum device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        # Iterate over 2x2 patches
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
                patches.append(measurement.view(bsz, self.n_wires))
        # Concatenate all patch measurements: (batch, 14*14, n_wires)
        measurements = torch.cat(patches, dim=1)
        # For each patch, apply heads and average across patches
        logits = self.classifier(measurements).mean(dim=1)
        regression = self.regressor(measurements).mean(dim=1).squeeze(-1)
        return {
            "logits": F.log_softmax(logits, dim=-1),
            "regression": regression,
        }

__all__ = ["HybridQuanvolutionRegressionModel"]
