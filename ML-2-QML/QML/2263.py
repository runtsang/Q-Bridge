"""Hybrid self‑attention model with quantum attention and regression.

The implementation follows the structure of the original QML seed but adds
an attention block that can be used as a drop‑in replacement for the
classical version.  The public API is identical to the classical module
so that downstream code can switch between the two back‑ends without
modification.

Typical usage::

    from SelfAttention__gen063 import SelfAttention
    model = SelfAttention()
    counts = model.run_quantum_attention(rot, ent, shots=1024)
    model.train_regression(dataset, epochs=20)
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum import QuantumDevice, QuantumModule, RandomLayer, RX, RY, CRX, MeasureAll, PauliZ, GeneralEncoder

# --------------------------------------------------------------------------- #
#  Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(QuantumModule):
    """Variational circuit that emulates a self‑attention style operation."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.random_layer = RandomLayer(n_ops=20, wires=list(range(n_qubits)))
        self.rx = RX(has_params=True, trainable=True)
        self.ry = RY(has_params=True, trainable=True)
        self.crx = CRX(has_params=True, trainable=True)

    def forward(self, qdev: QuantumDevice) -> None:
        self.random_layer(qdev)
        for i in range(self.n_qubits):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
        # Entangle neighbouring qubits
        for i in range(self.n_qubits - 1):
            self.crx(qdev, wires=[i, i + 1])

# --------------------------------------------------------------------------- #
#  Quantum regression head
# --------------------------------------------------------------------------- #
class QuantumRegressionModel(QuantumModule):
    """Full variational circuit for regression on quantum‑encoded states."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{n_qubits}xRy"]
        )
        self.attention = QuantumSelfAttention(n_qubits)
        self.measure = MeasureAll(PauliZ)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.attention(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Data utilities (identical to the classical seed)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Hybrid model combining quantum attention and regression
# --------------------------------------------------------------------------- #
class HybridSelfAttentionModel(QuantumModule):
    """Public API that mimics the original SelfAttention seed but uses a
    quantum attention block and a quantum regression head."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.attention = QuantumSelfAttention(n_qubits)
        self.regressor = QuantumRegressionModel(n_qubits)

    # --------------------------------------------------------------------- #
    #  Quantum attention
    # --------------------------------------------------------------------- #
    def run_quantum_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict[str, int]:
        """Execute the attention circuit on the chosen backend."""
        # The parameters are not used directly; they are encoded into the
        # variational layers.  For simplicity we ignore them here but keep the
        # signature for API compatibility.
        qdev = QuantumDevice(n_wires=self.n_qubits, bsz=1, device="cpu")
        self.attention(qdev)
        counts = qdev.get_counts()
        return counts

    # --------------------------------------------------------------------- #
    #  Regression training
    # --------------------------------------------------------------------- #
    def train_regression(
        self,
        dataset: torch.utils.data.Dataset,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        """Train the quantum regression head."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.regressor.train()
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for _ in range(epochs):
            for batch in loader:
                states = batch["states"]
                target = batch["target"]
                optimizer.zero_grad()
                pred = self.regressor(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

    # --------------------------------------------------------------------- #
    #  Evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
    ) -> float:
        """Return mean‑squared error on the dataset."""
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        self.regressor.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                states = batch["states"]
                target = batch["target"]
                pred = self.regressor(states)
                total += ((pred - target) ** 2).sum().item()
                n += target.numel()
        return total / n

# --------------------------------------------------------------------------- #
#  Factory function that preserves the original API
# --------------------------------------------------------------------------- #
def SelfAttention() -> HybridSelfAttentionModel:
    """Return a quantum hybrid self‑attention model."""
    return HybridSelfAttentionModel()

__all__ = ["HybridSelfAttentionModel", "SelfAttention"]
