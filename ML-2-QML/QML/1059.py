import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

dev = qml.device("default.qubit", wires=4)

def quantum_circuit(patch, weights):
    for i in range(weights.shape[0]):
        qml.RY(patch[0] * weights[i,0], wires=0)
        qml.RY(patch[1] * weights[i,1], wires=1)
        qml.RY(patch[2] * weights[i,2], wires=2)
        qml.RY(patch[3] * weights[i,3], wires=3)
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[2,3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

class QuanvolutionFilter(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.randn(n_layers, 4))
        @qml.qnode(dev, interface="torch")
        def qnode(patch_tensor, weights_tensor):
            return quantum_circuit(patch_tensor, weights_tensor)
        self.qnode = qnode
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].view(bsz, 4)
                qfeat = torch.stack([self.qnode(patch[i], self.weights) for i in range(bsz)], dim=0)
                feat = qfeat + patch
                patches.append(feat)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
