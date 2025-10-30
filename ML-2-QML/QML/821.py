import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = qml.device("default.qubit", wires=4)

class VariationalLayer(nn.Module):
    """Parameter‑efficient variational circuit with adaptive rotations."""
    def __init__(self, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, 4, 3))

    def forward(self, x):
        # x shape: (batch, 4)
        batch = x.shape[0]
        out = torch.zeros(batch, 4, device=x.device)
        for i in range(batch):
            out[i] = self._circuit(x[i], self.params[i])
        return out

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def _circuit(self, vec, params):
        # Amplitude encoding
        qml.AmplitudeEmbedding(features=vec, wires=range(4))
        for layer in range(self.n_layers):
            for w in range(4):
                qml.RY(params[layer, w, 0], wires=w)
                qml.RZ(params[layer, w, 1], wires=w)
            # Entangling pattern
            for w in range(3):
                qml.CNOT(wires=[w, w+1])
            qml.CNOT(wires=[3, 0])
        return [qml.expval(qml.PauliZ(w)) for w in range(4)]

class QFCModel(nn.Module):
    """Quantum model using a variational circuit with adaptive measurement."""
    def __init__(self):
        super().__init__()
        self.variational = VariationalLayer()
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz = x.shape[0]
        # Reduce to 4‑dim vector per sample
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        features = pooled[:, :4]
        out = self.variational(features)
        return self.norm(out)

__all__ = ["QFCModel"]
