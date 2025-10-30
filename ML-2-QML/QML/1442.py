import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate superposition states |ψ(θ,φ)⟩ = cosθ|0...0⟩ + e^{iφ}sinθ|1...1⟩
    with a richer target function.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(3 * thetas) * np.cos(2 * phis) + 0.05 * np.random.randn(samples)
    return states, labels

class RegressionDataset(Dataset):
    """Quantum regression dataset with complex amplitudes."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class QuantumFeatureMap(nn.Module):
    """Data‑re‑uploading feature map with entanglement."""
    def __init__(self, num_wires: int, depth: int = 2):
        super().__init__()
        self.num_wires = num_wires
        self.depth = depth
        self.dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            for d in range(self.depth):
                for wire in range(num_wires):
                    qml.RX(x[wire], wires=wire)
                    qml.RY(x[wire], wires=wire)
                for wire in range(num_wires-1):
                    qml.CNOT(wires=[wire, wire+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.circuit(x)

class QModel(nn.Module):
    """Hybrid quantum‑classical regression model."""
    def __init__(self, num_wires: int, device: str = "cpu"):
        super().__init__()
        self.feature_map = QuantumFeatureMap(num_wires, depth=3)
        self.head = nn.Linear(num_wires, 1)
        self.device = device

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Convert complex amplitudes to phases for the feature map
        angles = torch.angle(state_batch)
        angles = angles[:, :self.feature_map.num_wires]
        features = self.feature_map(angles.to(self.device))
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
