import torch
from torch import nn
import torch.nn.functional as F

class ConvGen512(nn.Module):
    """
    Hybrid convolutional generator that fuses a classical conv branch
    with an optional quantum feature extractor. The module outputs
    512-dimensional logits suitable for classification or embedding.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 hidden_dim: int = 256,
                 n_qubits: int = 0,
                 threshold: float = 0.0,
                 use_quantum: bool = True,
                 fusion_weight: float = 0.5,
                 device: torch.device = torch.device("cpu")):
        super().__init__()
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.device = device

        # Classical conv branch
        self.conv = nn.Conv2d(1, hidden_dim, kernel_size=kernel_size, bias=True)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # Fusion linear layer
        self.fusion = nn.Linear(hidden_dim * (kernel_size ** 2) + 1, 512)

        # Optional quantum branch
        if self.use_quantum and self.n_qubits > 0:
            from qml_code import ConvQuantum
            self.quantum = ConvQuantum(kernel_size=kernel_size,
                                       threshold=threshold,
                                       device=device)
        else:
            self.quantum = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, H, W) where H and W are multiples
               of kernel_size.

        Returns:
            logits: Tensor of shape (batch_size, 512)
        """
        batch_size = x.size(0)
        # Extract patches
        patches = x.unfold(2, self.kernel_size, self.kernel_size).unfold(3, self.kernel_size, self.kernel_size)
        num_patches = patches.shape[2] * patches.shape[3]
        patches = patches.contiguous().view(-1, 1, self.kernel_size, self.kernel_size)
        conv_out = self.conv(patches)  # (batch_size * num_patches, hidden_dim, 1, 1)
        conv_out = conv_out.view(batch_size, num_patches, self.hidden_dim)
        conv_out = self.bn(conv_out)
        conv_out = self.relu(conv_out)
        conv_feat = conv_out.view(batch_size, -1)  # (batch_size, hidden_dim * num_patches)

        # Quantum branch
        if self.quantum is not None:
            quantum_input = patches.view(batch_size * num_patches, self.kernel_size, self.kernel_size)
            quantum_output = self.quantum(quantum_input)  # (batch_size * num_patches,)
            quantum_output = quantum_output.view(batch_size, num_patches)
            quantum_feat = quantum_output.mean(dim=1, keepdim=True)  # (batch_size, 1)
        else:
            quantum_feat = torch.zeros(batch_size, 1, device=self.device)

        fused = torch.cat([conv_feat, quantum_feat], dim=1)
        logits = self.fusion(fused)
        return logits
