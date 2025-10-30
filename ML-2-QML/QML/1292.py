import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumNATEnhanced(nn.Module):
    """
    Hybrid classical‑quantum model that encodes pooled image patches into a 4‑wire qubit system,
    applies a two‑layer strongly entangling variational circuit, and returns the
    expectation values of Pauli‑Z on each wire as a 4‑dimensional classification vector.
    """
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.wires = 4
        # Parameter matrix for the variational circuit: shape (num_layers, wires, 3)
        self.params = nn.Parameter(torch.randn(num_layers, self.wires, 3))
        self.device = qml.device("default.qubit", wires=self.wires, shots=None)
        # QNode that performs amplitude embedding followed by a strongly entangling layer
        self.circuit = qml.qnode(self.device, interface="torch")(self._circuit)
        self.norm = nn.BatchNorm1d(self.wires)

    @staticmethod
    def _circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        x: 1‑D tensor of length 2^wires (16 for 4 wires) – the amplitude‑encoded input.
        params: trainable parameters for the variational circuit.
        """
        # Amplitude embedding (normalised)
        qml.AmplitudeEmbedding(features=x, wires=range(4), normalize=True)
        # Strongly entangling layers
        qml.templates.StronglyEntanglingLayers(params, wires=range(4))
        # Return expectation values of Pauli‑Z on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: average‑pool the input to 4×4, flatten to 16‑dim, encode, and evaluate the
        variational circuit. The output is a (batch, 4) tensor of expectation values.
        """
        # Average‑pool to 4×4 (kernel size 6 reduces 28→4)
        pooled = F.avg_pool2d(x, kernel_size=6).view(x.shape[0], -1)  # shape (batch, 16)
        # Evaluate the circuit for each sample in the batch
        # Pennylane supports batched inputs when shots=None; otherwise loop over batch
        if pooled.ndim == 2:
            # Batched execution
            out = torch.stack([self.circuit(sample, self.params) for sample in pooled], dim=0)
        else:
            out = self.circuit(pooled, self.params)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
