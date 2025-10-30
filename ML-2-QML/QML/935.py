import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionGen216(nn.Module):
    """Quantum variant: 2×2 patches processed by a variational two‑qubit ansatz."""
    def __init__(self, device='default.qubit', wires=4, n_layers=3):
        super().__init__()
        self.device = device
        self.wires = wires
        self.n_layers = n_layers
        self.qdevice = qml.device(self.device, wires=self.wires)
        self.q_params = nn.Parameter(torch.randn(n_layers, wires))
        self.linear = nn.Linear(4 * 14 * 14, 10)

        @qml.qnode(self.qdevice, interface='torch', diff_method='backprop')
        def circuit(x, params):
            for i in range(self.n_layers):
                for w in range(self.wires):
                    qml.RX(x[w], wires=w)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[2, 3])
                for w in range(self.wires):
                    qml.RZ(params[i, w], wires=w)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.wires)]

        self.circuit = circuit

    def forward(self, x):
        batch = x.shape[0]
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r:r+2, c:c+2].view(batch, -1)
                out = torch.stack([self.circuit(patch[i], self.q_params) for i in range(batch)], dim=0)
                patches.append(out)
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen216"]
