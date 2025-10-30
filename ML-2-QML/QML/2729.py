"""Hybrid quantum model that encodes images and quantum states for classification and regression."""
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from torch.nn.functional import avg_pool2d

class HybridNATRegressor(tq.QuantumModule):
    """
    Quantum counterpart to HybridNATRegressor.
    Uses a shared encoder to transform either image patches or raw quantum states
    into a set of qubit amplitudes. Two parallel QLayers produce features for
    classification and regression respectively.
    """
    class QLayer(tq.QuantumModule):
        """Reusable variational layer with random gates and parameterized rotations."""
        def __init__(self, n_wires: int, n_ops: int = 30):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_classes: int = 4, regression: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.regression = regression

        # Image encoder: 4‑qubit 4×4_ryzxy circuit
        self.image_encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        # State encoder: n‑qubit Ry circuit
        self.n_wires_state = 4
        self.state_encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.n_wires_state}xRy"]
        )

        self.class_layer = self.QLayer(self.n_wires_state)
        self.reg_layer = self.QLayer(self.n_wires_state) if regression else None

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.class_head = nn.Linear(self.n_wires_state, num_classes)
        if regression:
            self.reg_head = nn.Linear(self.n_wires_state, 1)

        self.class_norm = nn.BatchNorm1d(num_classes)
        if regression:
            self.reg_norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Accepts either a batch of 2‑D images (b,1,28,28) or a batch of quantum state vectors.
        The tensor shape determines the encoding path.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires_state, bsz=bsz, device=x.device)

        if x.ndim == 4:  # image batch
            # Average‑pool to 6×6 then flatten to 16 features
            pooled = avg_pool2d(x, 6).view(bsz, 16)
            self.image_encoder(qdev, pooled)
        else:  # raw state vectors
            self.state_encoder(qdev, x)

        # Classification branch
        self.class_layer(qdev)
        class_feats = self.measure(qdev)
        class_out = self.class_head(class_feats)
        out = {"class": self.class_norm(class_out)}

        if self.regression:
            # Regression branch
            self.reg_layer(qdev)
            reg_feats = self.measure(qdev)
            reg_out = self.reg_head(reg_feats).squeeze(-1)
            out["regress"] = self.reg_norm(reg_out)

        return out

# Data utilities --------------------------------------------------------------

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates quantum states |ψ⟩ = cosθ|0…0⟩ + e^{iφ} sinθ|1…1⟩
    and corresponding regression targets.
    """
    omega0 = np.zeros(2 ** num_wires, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(2 ** num_wires, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(tq.QuantumModule):
    """
    Dataset for quantum regression task.
    """
    def __init__(self, samples: int, num_wires: int):
        super().__init__()
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["HybridNATRegressor", "RegressionDataset", "generate_superposition_data"]
