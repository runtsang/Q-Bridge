import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumLayer(tq.QuantumModule):
    """Parameterised quantum layer with randomised gates and trainable rotations."""
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

class HybridQuantumHead(tq.QuantumModule):
    """Quantum head that encodes classical activations and produces a scalar logit."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = QuantumLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

class HybridQCNet(nn.Module):
    """Hybrid classical‑quantum binary classifier with a quantum head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.quantum_head = HybridQuantumHead(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = self.quantum_head(x)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["HybridQCNet"]
