import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities – generate classical features and quantum state samples
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a (features, labels) pair where features are real-valued and
    labels are derived from a sinusoidal function of the summed features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class HybridRegressionDataset(Dataset):
    """Dataset that exposes both classical features and their corresponding
    quantum states (encoded as cos(theta)|0...0>+e^{i phi}sin(theta)|1...1>).
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
# Classical + Quantum hybrid model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(nn.Module):
    """
    A hybrid regression network that combines a classical MLP with a
    quantum feature extractor.  The user can toggle `use_quantum` to
    experiment with purely classical, purely quantum, or hybrid training.
    """
    def __init__(self,
                 num_features: int,
                 num_wires: int,
                 use_quantum: bool = True):
        super().__init__()
        self.use_quantum = use_quantum
        self.num_features = num_features
        self.num_wires = num_wires

        # Classical MLP
        self.classical_net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # Quantum head – only registered if quantum module is available
        if self.use_quantum:
            try:
                import torchquantum as tq
            except Exception as e:  # pragma: no cover
                raise RuntimeError("torchquantum is required for the quantum head") from e
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
            )
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            # Final linear head fuses classical and quantum features
            self.final_head = nn.Linear(8 + num_wires, 1)
        else:
            self.final_head = nn.Linear(8, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Accepts a batch with keys `features` and optionally `states`."""
        x = batch["features"]
        # Classical branch
        c_out = self.classical_net(x)

        if self.use_quantum:
            # Prepare quantum device
            bsz = x.shape[0]
            device = x.device
            qdev = self.encoder.qdev if hasattr(self.encoder, "qdev") else None
            if qdev is None:
                import torchquantum as tq
                qdev = tq.QuantumDevice(self.num_wires, bsz=bsz, device=device)
            # Encode classical data
            self.encoder(qdev, x)
            # Random feature layer
            self.random_layer(qdev)
            # Measurement
            q_out = self.measure(qdev)
            # Concatenate classical + quantum features
            combined = torch.cat([c_out, q_out], dim=-1)
        else:
            combined = c_out

        return self.final_head(combined).squeeze(-1)


# --------------------------------------------------------------------------- #
# Sampler QNN – a lightweight probability output module
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """
    Produces a soft‑max distribution over two outcomes.  Useful for
    probabilistic regression or classification experiments.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)


# --------------------------------------------------------------------------- #
# Quanvolution filter – quantum‑aware image patch extractor
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """
    Applies a random 2‑qubit quantum kernel to every 2x2 patch of a 28x28 image.
    The output is a flattened feature vector ready for a linear classifier.
    """
    def __init__(self, num_wires: int = 4):
        super().__init__()
        try:
            import torchquantum as tq
        except Exception as e:  # pragma: no cover
            raise RuntimeError("torchquantum is required for QuanvolutionFilter") from e
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(num_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = self.encoder.qdev if hasattr(self.encoder, "qdev") else None
        if qdev is None:
            import torchquantum as tq
            qdev = tq.QuantumDevice(self.num_wires, bsz=bsz, device=device)

        # Reshape to 28x28 grayscale
        img = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        img[:, r, c],
                        img[:, r, c + 1],
                        img[:, r + 1, c],
                        img[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, self.num_wires))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
__all__ = [
    "HybridRegressionDataset",
    "HybridRegressionModel",
    "SamplerQNN",
    "QuanvolutionFilter",
]
