import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import torch.nn.functional as F

class QuantumHybridNAT(tq.QuantumModule):
    """
    Hybrid model that uses a classical convolutional backbone,
    a quantum kernel applied to image patches, and a quantum
    fully‑connected head that maps aggregated quantum features
    to logits, followed by a classical linear classifier.
    """

    class QKernel(tq.QuantumModule):
        """
        Quantum kernel applied to a 2×2 patch (4 input values).
        """
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Encode each patch value into a rotation on a distinct qubit
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
            """
            Parameters
            ----------
            x : torch.Tensor
                Patch tensor of shape (batch_size, 4).

            Returns
            -------
            torch.Tensor
                Measurement vector of shape (batch_size, n_wires).
            """
            bsz = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            self.encoder(qdev, x)
            self.random_layer(qdev)
            measurement = self.measure(qdev)
            return measurement

    class QFCHead(tq.QuantumModule):
        """
        Quantum fully‑connected head that maps aggregated
        quantum features to a 4‑dimensional vector.
        """
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.ry = tq.RY(has_params=True, trainable=True)
            self.cx = tq.CX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Aggregated quantum features of shape (batch_size, n_wires).

            Returns
            -------
            torch.Tensor
                Normalized vector of shape (batch_size, n_wires).
            """
            bsz = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            self.ry(qdev, wires=range(self.n_wires), params=x)
            self.cx(qdev, wires=[0, 1])
            self.cx(qdev, wires=[2, 3])
            out = self.measure(qdev)
            return self.norm(out)

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.n_wires = 4
        # Classical convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.qkernel = self.QKernel()
        self.qhead = self.QFCHead(n_wires=self.n_wires)
        self.classifier = nn.Linear(self.n_wires, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, n_classes).
        """
        bsz = x.shape[0]
        features = self.backbone(x)  # (bsz, 16, 7, 7)
        # Reshape to a sequence of 4‑element patches
        patches = features.view(bsz, 16 * 7 * 7)
        n_patches = patches.shape[1] // 4
        patch_features = []
        for i in range(n_patches):
            patch = patches[:, i * 4 : (i + 1) * 4]  # (bsz, 4)
            feat = self.qkernel(patch)              # (bsz, 4)
            patch_features.append(feat)
        # Stack and average across all patches to obtain a global feature vector
        quantum_features = torch.stack(patch_features, dim=1)  # (bsz, n_patches, 4)
        aggregated = quantum_features.mean(dim=1)             # (bsz, 4)
        # Pass through the quantum fully‑connected head
        quantum_out = self.qhead(aggregated)                 # (bsz, 4)
        # Final classification
        logits = self.classifier(quantum_out)                 # (bsz, n_classes)
        return logits

__all__ = ["QuantumHybridNAT"]
