"""
HybridQuantumCNN: A quantum-augmented network that combines a QCNN-style feature
map, a quantum transformer block, and a quantum kernel for classification.
The architecture is inspired by QCNN, GraphQNN, QTransformerTorch, and
QuantumKernelMethod, providing a fully quantum-capable baseline.
"""

import torch
import torch.nn as nn
from QTransformerTorch import TransformerBlockQuantum
from QuantumKernelMethod import Kernel

class HybridQuantumCNN(nn.Module):
    """
    Quantum-enhanced version of HybridQuantumCNN. It encodes input data into a
    quantum state via a QCNN-style feature map, processes the resulting state
    with a quantum transformer block, and finally classifies using a quantum
    kernel against a set of learnable prototype vectors.
    """

    def __init__(
        self,
        num_classes: int = 2,
        transformer_blocks: int = 2,
        ffn_dim: int = 64,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 4,
        n_qlayers: int = 1,
        num_prototypes: int = 10,
    ) -> None:
        super().__init__()
        # Learnable prototypes for kernel classification
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 8))
        # Quantum kernel (from QuantumKernelMethod)
        self.kernel = Kernel()
        # Quantum transformer block
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim=num_prototypes,
                    num_heads=2,
                    ffn_dim=ffn_dim,
                    n_qubits_transformer=n_qubits_transformer,
                    n_qubits_ffn=n_qubits_ffn,
                    n_qlayers=n_qlayers,
                )
                for _ in range(transformer_blocks)
            ]
        )
        # Classifier
        self.classifier = nn.Linear(num_prototypes, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 8)
        batch_size = x.size(0)
        kernel_matrix = []
        # Compute quantum kernel between each input and each prototype
        for i in range(batch_size):
            row = []
            for proto in self.prototypes:
                k = self.kernel(x[i].unsqueeze(0), proto.unsqueeze(0)).squeeze()
                row.append(k)
            kernel_matrix.append(torch.stack(row))
        kernel_matrix = torch.stack(kernel_matrix)  # (batch, num_prototypes)
        # Prepare for transformer: sequence length 1
        seq = kernel_matrix.unsqueeze(1)  # (batch, 1, num_prototypes)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)
        out = self.classifier(seq)
        return torch.sigmoid(out) if out.shape[-1] == 1 else out

__all__ = ["HybridQuantumCNN"]
