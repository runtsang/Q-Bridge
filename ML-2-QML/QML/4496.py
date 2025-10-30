import torch
import torch.nn as nn
import torchquantum as tq
from typing import List, Iterable, Callable

class HybridQuanvolutionClassifier(tq.QuantumModule):
    """Quantum counterpart of the hybrid quanvolution model.

    Features:
    * Two‑qubit patch encoding via Ry rotations.
    * Random layer and trainable RX/RY per qubit.
    * Measurement in Z basis to produce classical features.
    * Linear head for classification.
    * FastEstimator‑style evaluation with optional shot noise.
    """

    class _PatchEncoder(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "ry", "wires": [0]},
                    {"input_idx": [1], "func": "ry", "wires": [1]},
                    {"input_idx": [2], "func": "ry", "wires": [2]},
                    {"input_idx": [3], "func": "ry", "wires": [3]},
                ]
            )
            self.random = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))

        def forward(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> None:
            self.encoder(qdev, patch)
            self.random(qdev)

    class _FeatureLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.n_wires = 4
        self.patch_encoder = self._PatchEncoder(self.n_wires)
        self.feature_layer = self._FeatureLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(self.n_wires, num_classes)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
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
                self.patch_encoder(qdev, patch)
                self.feature_layer(qdev)
                measure = self.measure(qdev)
                patches.append(measure.view(bsz, self.n_wires))
        features = torch.cat(patches, dim=1)
        return features

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward(x)
        logits = self.classifier(features)
        return torch.log_softmax(logits, dim=-1)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return quantum kernel matrix between two batches of images."""
        a_feat = self.forward(a)
        b_feat = self.forward(b)
        diff = a_feat.unsqueeze(1) - b_feat.unsqueeze(0)
        dist_sq = (diff * diff).sum(dim=-1)
        return torch.exp(-self.gamma * dist_sq)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        parameter_sets: List[List[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Evaluate observables on batches of parameters with optional shot noise."""
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.classify(inputs)
                row = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = torch.Generator()
        if seed is not None:
            rng.manual_seed(seed)
        noisy = []
        for row in results:
            noisy_row = [float(torch.normal(mean, max(1e-6, 1 / shots), generator=rng)) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQuanvolutionClassifier"]
