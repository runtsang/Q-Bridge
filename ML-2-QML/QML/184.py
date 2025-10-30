import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="torch")
def quantum_circuit(theta: torch.Tensor) -> torch.Tensor:
    """Simple parameterised quantum circuit returning ⟨Z⟩."""
    qml.Hadamard(0)
    qml.RY(theta, 0)
    return qml.expval(qml.PauliZ(0))

class QuantumHybridClassifier(nn.Module):
    """CNN + attention head followed by a quantum expectation layer."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.attn = nn.MultiheadAttention(embed_dim=120, num_heads=4, dropout=0.1)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # Prepare for attention: (seq_len, batch, embed_dim)
        x = x.unsqueeze(0)  # seq_len=1
        attn_output, _ = self.attn(x, x, x)
        x = attn_output.squeeze(0)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Quantum head
        theta = x.squeeze(-1)  # shape (B,)
        q_output = torch.stack([quantum_circuit(t) for t in theta])
        probs = torch.sigmoid(q_output)
        return torch.cat((probs, 1 - probs), dim=-1)
