import pennylane as qml
import torch
import torch.nn as nn

# Quantum device: 4 qubits
dev = qml.device('default.qubit', wires=4)

def _qnode(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Variational circuit that receives a 4‑dimensional real vector x."""
    # Encode the classical input onto the qubits
    for i in range(4):
        qml.RY(x[i], wires=i)
    # Variational layers
    for layer in range(params.shape[0]):
        qml.RX(params[layer, 0], wires=0)
        qml.RX(params[layer, 1], wires=1)
        qml.RX(params[layer, 2], wires=2)
        qml.RX(params[layer, 3], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

# Pennylane qnode with Torch interface for autograd
qnode = qml.qnode(dev, interface='torch', diff_method='backprop')

class QFCModel(nn.Module):
    """Quantum neural network with a parameter‑shuffled variational ansatz."""
    def __init__(self, n_layers: int = 3) -> None:
        super().__init__()
        self.n_layers = n_layers
        # Trainable parameters: [n_layers, 4] (one RX per wire per layer)
        self.params = nn.Parameter(torch.randn(n_layers, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, 1, 28, 28)
        Returns: Tensor of shape (batch, 4) after quantum processing and normalisation.
        """
        batch_size = x.size(0)
        out = []
        for idx in range(batch_size):
            # Adaptive average pooling to 6×6, then flatten to 16 features
            pooled = torch.nn.functional.avg_pool2d(x[idx:idx+1], kernel_size=6).view(-1)
            # Ensure the vector has length 4 (e.g., by linear projection)
            if pooled.shape[0]!= 4:
                pooled = nn.functional.linear(pooled, torch.randn(4, pooled.shape[0], device=x.device))
            out.append(qnode(self.params, pooled))
        out = torch.stack(out)
        return self.norm(out)
