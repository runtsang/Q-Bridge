import pennylane as qml
import torch
import numpy as np
from torch import nn
from pennylane import numpy as pnp

class VariationalQuanvolutionCircuit(nn.Module):
    """
    Variational circuit that maps a 4‑dimensional classical patch to a
    4‑dimensional quantum measurement vector.
    """
    def __init__(self,
                 num_qubits: int = 4,
                 num_layers: int = 3,
                 device: str = "default.qubit",
                 shots: int = 0) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits, shots=shots)

        # Randomly initialise trainable parameters
        self.params = nn.Parameter(
            torch.randn(num_layers, num_qubits, 3) * 0.1,  # rotations about x, y, z
            requires_grad=True
        )

        # Build a batched QNode that processes many patches simultaneously
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # inputs: (batch, 4) where each element ∈ [0,1]
            # params: (num_layers, num_qubits, 3)
            # Use qml.map to vectorise over batch dimension
            def _apply_one_patch(inp):
                for qubit in range(num_qubits):
                    qml.RX(inp[qubit], wires=qubit)
                for l in range(num_layers):
                    for qubit in range(num_qubits):
                        rot = params[l, qubit]
                        qml.Rot(rot[0], rot[1], rot[2], wires=qubit)
                    # Entanglement layer
                    for q in range(num_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

            return qml.map(_apply_one_patch, inputs)

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, N, 4)
            Batch of patches (N per image, 4 features each).

        Returns
        -------
        torch.Tensor
            Quantum measurements of shape (B, N, 4).
        """
        return self._circuit(x, self.params)

class QuanvolutionGen357QuantumClassifier(nn.Module):
    """
    Subclass of the classical classifier that plugs in the variational
    quantum circuit.  The quantum layer outputs a 4‑dimensional vector per patch.
    """
    def __init__(self,
                 num_classes: int = 10,
                 patch_size: int = 2,
                 in_channels: int = 1,
                 head_dim: int = 4,
                 num_heads: int = 3,
                 device: str = "default.qubit",
                 shots: int = 0) -> None:
        super().__init__()
        # Reuse classical filter from the ML module
        self.qfilter = MultiHeadQuanvolutionFilter(
            patch_size=patch_size,
            in_channels=in_channels,
            head_dim=head_dim,
            num_heads=num_heads
        )
        self.quantum_layer = VariationalQuanvolutionCircuit(
            num_qubits=head_dim,
            num_layers=3,
            device=device,
            shots=shots
        )
        self.linear = nn.Linear(head_dim * 14 * 14 * num_heads, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)                     # (B, H*4, 14, 14)
        B, H, H_out, W_out = features.shape
        features = features.view(B, H * H_out * W_out, 4)   # (B, N, 4)
        q_features = self.quantum_layer(features)          # (B, N, 4)
        q_features = q_features.view(B, -1)                # (B, 4*H*14*14)
        logits = self.linear(q_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["VariationalQuanvolutionCircuit", "QuanvolutionGen357QuantumClassifier"]
