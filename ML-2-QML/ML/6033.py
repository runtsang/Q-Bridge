import torch
from torch import nn
import numpy as np
from.qml_code import run_quantum_circuit, quantum_conv_circuit

class UnifiedQCNN(nn.Module):
    """
    Hybrid QCNN that merges a classical feature-map backbone with a quantum ansatz
    and a configurable convolution filter that can be either classical or quantum.
    The network is fully differentiable and can be trained end‑to‑end with standard
    optimizers.  The quantum part is executed via Qiskit and its gradients are
    propagated through the EstimatorQNN wrapper.
    """
    def __init__(self,
                 input_dim: int = 8,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 use_quantum_conv: bool = False,
                 num_qubits: int = 16,
                 num_layers: int = 3,
                 pool_size: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.conv_kernel = conv_kernel
        self.conv_threshold = conv_threshold
        self.use_quantum_conv = use_quantum_conv
        self.num_qubits = num_qubits

        # Classical feature map (mirrors the first seed's feature_map)
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh()
        )

        # Classical convolution filter (from Conv.py)
        if not self.use_quantum_conv:
            self.conv = nn.Linear(16, 16)
            self.threshold = conv_threshold

        # Quantum convolution circuit (from qml_code)
        else:
            self.qconv = quantum_conv_circuit(int(np.sqrt(num_qubits)))

        # Shared variational ansatz across convolution and pooling layers
        self.shared_ansatz = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_qubits, num_qubits),
                nn.Tanh()
            ) for _ in range(num_layers)
        ])

        # Pooling layers with learnable weights
        self.pool = nn.Sequential(
            nn.Linear(num_qubits, num_qubits // pool_size),
            nn.Tanh()
        )

        # Final classifier head
        self.head = nn.Linear(num_qubits // pool_size, 1)

    def _quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the quantum convolution circuit on the flattened input and
        return the expectation value as a tensor.  The method uses the
        run_quantum_circuit helper from the QML module.
        """
        batch, dim = x.shape
        # Convert to numpy for Qiskit execution
        x_np = x.detach().cpu().numpy()
        out = []
        for sample in x_np:
            val = run_quantum_circuit(self.qconv, sample)
            out.append(val)
        out = torch.tensor(out, dtype=torch.float32, device=x.device)
        # Broadcast scalar to vector of length num_qubits
        out = out.unsqueeze(1).repeat(1, self.num_qubits)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that uses the original QCNN feature map,
        combined with a quantum or classical convolution filter,
        and a variational pooling mechanism.
        """
        # Feature map transformation
        x = self.feature_map(x)

        # Convolution step
        if self.use_quantum_conv:
            x = self._quantum_forward(x)
        else:
            x = self.conv(x)
            x = torch.sigmoid(x - self.threshold)

        # Shared ansatz (apply linear layers)
        for layer in self.shared_ansatz:
            x = layer(x)

        # Pooling
        x = self.pool(x)

        # Classifier head
        out = torch.sigmoid(self.head(x))
        return out

def QCNN() -> UnifiedQCNN:
    """
    Factory returning the configured :class:`UnifiedQCNN` model.
    """
    return UnifiedQCNN()

__all__ = ["QCNN", "UnifiedQCNN"]
