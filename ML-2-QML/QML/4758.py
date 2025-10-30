import math
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dataset of superposition states |psi> = cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    Returns state tensor (samples, 2**num_wires) and target tensor (samples,).
    """
    dim = 2 ** num_wires
    omega_0 = torch.zeros(dim, dtype=torch.complex64)
    omega_0[0] = 1.0
    omega_1 = torch.zeros(dim, dtype=torch.complex64)
    omega_1[-1] = 1.0

    thetas = 2 * math.pi * torch.rand(samples)
    phis = 2 * math.pi * torch.rand(samples)
    states = torch.zeros((samples, dim), dtype=torch.complex64)
    for i in range(samples):
        states[i] = torch.cos(thetas[i]) * omega_0 + torch.exp(1j * phis[i]) * torch.sin(thetas[i]) * omega_1

    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset for quantum regression with complex states.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # pragma: no cover
        return self.states.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # pragma: no cover
        return {
            "states": self.states[idx],
            "target": self.labels[idx],
        }


class ScaleShiftQuantum(nn.Module):
    """
    Learnable scale and shift applied to the measurement output.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class HybridQuantumRegression(tq.QuantumModule):
    """
    Hybrid quantum regression model combining a random layer, RX/RY rotations, and a classical head.
    """
    def __init__(
        self,
        num_wires: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.scale_shift = ScaleShiftQuantum()
        self.head = nn.Linear(num_wires, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.random_layer(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        features = self.measure(qdev)
        features = self.dropout(features)
        features = self.scale_shift(features)
        return self.head(features).squeeze(-1)


__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "HybridQuantumRegression",
]
