import torch
import torch.nn as nn
import torchquantum as tq

class QLayer(tq.QuantumModule):
    """Variational layer using a random circuit followed by trainable rotations."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice):
        self.random(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

class HybridBinaryClassifier(tq.QuantumModule):
    """End‑to‑end quantum binary classifier with a feature encoder and a variational layer."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical data into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.qlayer = QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.qlayer(qdev)
        features = self.measure(qdev)
        probs = torch.sigmoid(self.head(features).squeeze(-1))
        return torch.stack([probs, 1 - probs], dim=-1)

def generate_superposition_data(num_wires: int, samples: int):
    """Generate complex superposition states and a sinusoidal target."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * np.eye(1, 2 ** num_wires, 0)[0] + \
                    np.exp(1j * phis[i]) * np.eye(1, 2 ** num_wires, -1)[0]
    labels = np.sin(2 * thetas) * np.cos(phis)
    labels = (labels > 0).astype(np.float32)
    return states, labels

class SuperpositionDataset(torch.utils.data.Dataset):
    """Dataset of quantum superposition states with binary labels."""
    def __init__(self, samples: int, num_wires: int = 4):
        states, labels = generate_superposition_data(num_wires, samples)
        self.states = torch.tensor(states, dtype=torch.cfloat)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return {"states": self.states[idx], "target": self.labels[idx]}

__all__ = ["HybridBinaryClassifier", "SuperpositionDataset", "generate_superposition_data"]
