import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

class QuantumPatchEncoder(tq.QuantumModule):
    """Encodes 2×2 patches into a 4‑qubit state."""
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
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice, patches: torch.Tensor) -> torch.Tensor:
        # patches: (batch, n_patches, 4)
        batch, n_patches, _ = patches.shape
        features = []
        for i in range(n_patches):
            data = patches[:, i, :]
            self.encoder(qdev, data)
            self.random_layer(qdev)
            meas = self.measure(qdev)
            features.append(meas)
        return torch.cat(features, dim=1)

class QuantumFullyConnectedLayer(tq.QuantumModule):
    """Trainable fully‑connected quantum layer."""
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

class QuanvolutionHybrid(nn.Module):
    """Hybrid CNN + quantum layers model."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Classical feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum modules
        self.encoder = QuantumPatchEncoder()
        self.qfc = QuantumFullyConnectedLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(4)

        # Linear head
        self.classifier = nn.Linear(68, num_classes)  # 64 from patches + 4 from qfc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        x = self.features(x)          # (B, 16, 7, 7)
        B, C, H, W = x.shape
        patches = []
        for r in range(0, H, 2):
            for c in range(0, W, 2):
                patch = x[:, :, r:r+2, c:c+2]          # (B, 16, 2, 2)
                patch_mean = patch.mean(dim=1)          # (B, 2, 2)
                patch_flat = patch_mean.view(B, 4)      # (B, 4)
                patches.append(patch_flat)
        patches = torch.stack(patches, dim=1)          # (B, 16, 4)
        # Quantum encoding
        qdev = tq.QuantumDevice(n_wires=self.encoder.n_wires, bsz=B, device=x.device)
        quantum_features = self.encoder(qdev, patches)  # (B, 64)
        # Fully connected quantum layer
        self.qfc(qdev)
        qfc_out = self.measure(qdev)  # (B, 4)
        qfc_out = self.norm(qfc_out)
        # Concatenate classical and quantum features
        features = torch.cat([quantum_features, qfc_out], dim=1)  # (B, 68)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

class FastBaseEstimator:
    """Evaluate a PyTorch model on a set of parameter vectors."""
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: list,
        parameter_sets: list,
    ):
        self.model.eval()
        results = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: list,
        parameter_sets: list,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ):
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = [
    "QuantumPatchEncoder",
    "QuantumFullyConnectedLayer",
    "QuanvolutionHybrid",
    "FastBaseEstimator",
    "FastEstimator",
]
