import torch
from torch import nn
import torch.nn.functional as F
import pennylane as qml

class QuantumBlock(nn.Module):
    """Differentiable variational quantum circuit used as a filter."""
    def __init__(self, n_qubits: int, depth: int = 2, entanglement: str = 'full'):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.params = nn.Parameter(torch.randn(depth, n_qubits))
        self.dev = qml.device('default.qubit', wires=n_qubits)

        def circuit(inputs, params):
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            for d in range(depth):
                for i in range(n_qubits):
                    qml.RZ(params[d, i], wires=i)
                if entanglement == 'full':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                    qml.CNOT(wires=[n_qubits - 1, 0])
                elif entanglement == 'pair':
                    for i in range(0, n_qubits, 2):
                        qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))
        self.qnode = qml.QNode(circuit, self.dev, interface='torch')

    def forward(self, x: torch.Tensor):
        return self.qnode(x, self.params)

class ConvEnhanced(nn.Module):
    """Hybrid classical‑quantum convolution filter."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 n_layers: int = 1,
                 quantum_depth: int = 2,
                 quantum_entanglement: str = 'full'):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_layers = n_layers
        self.classical = nn.Sequential(
            *[nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
              for _ in range(n_layers)],
            nn.ReLU()
        )
        self.quantum_block = QuantumBlock(kernel_size**2,
                                          depth=quantum_depth,
                                          entanglement=quantum_entanglement)

    def forward(self, x: torch.Tensor):
        cls = self.classical(x)
        B, C, H, W = cls.shape
        patches = cls.unfold(2, self.kernel_size, self.kernel_size).unfold(3, self.kernel_size,
                                                                         self.kernel_size)
        patches = patches.contiguous().view(B, 1, -1, self.kernel_size**2)
        quantum_vals = []
        for i in range(patches.shape[2]):
            patch = patches[:, :, i].reshape(B, -1)
            quantum_vals.append(self.quantum_block(patch).unsqueeze(1))
        quantum_map = torch.cat(quantum_vals, dim=1)
        quantum_map = quantum_map.view(B, 1, H // self.kernel_size, W // self.kernel_size)
        quantum_map = F.interpolate(quantum_map, scale_factor=self.kernel_size,
                                    mode='nearest')
        out = torch.cat([cls, quantum_map], dim=1)
        return out
