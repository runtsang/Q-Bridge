import torch
from torch import nn
import numpy as np
from qml_module import QCNN, SelfAttention

class HybridQuantumConvolutionSelfAttention(nn.Module):
    """
    Hybrid neural network that combines:
    1. Classical convolution‑inspired feature extractor.
    2. Quantum convolutional neural network (QCNN) via EstimatorQNN.
    3. Quantum self‑attention block producing a probability distribution over qubit states.
    All components are trainable in a single end‑to‑end pipeline.
    """

    def __init__(self, input_dim: int = 8, num_qubits: int = 4):
        super().__init__()
        # Classical feature extractor
        self.classical_feature = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh()
        )
        # Quantum convolution (QCNN)
        self.quantum_cnn = QCNN()
        # Create torch parameters mirroring the ansatz weights
        self.quantum_params = nn.ParameterList(
            [nn.Parameter(torch.randn(p.shape)) for p in self.quantum_cnn.weight_params]
        )
        # Quantum self‑attention
        self.quantum_attention = SelfAttention()
        # Parameters for the attention circuit
        self.attention_rot_params = nn.Parameter(torch.randn(num_qubits * 3))
        self.attention_ent_params = nn.Parameter(torch.randn(num_qubits - 1))
        # Final classification head
        attention_dim = 2 ** num_qubits
        self.head = nn.Linear(16 + 1 + attention_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        x = self.classical_feature(inputs)
        # Quantum convolution output (shape: [batch, 1])
        q_out = self.quantum_cnn(x, weight_params=list(self.quantum_params)).unsqueeze(-1)
        # Quantum self‑attention output (shape: [batch, 2**num_qubits])
        attn_np = self.quantum_attention.run(
            rotation_params=self.attention_rot_params.detach().cpu().numpy(),
            entangle_params=self.attention_ent_params.detach().cpu().numpy()
        )
        attn = torch.tensor(attn_np, device=inputs.device, dtype=inputs.dtype)
        # Concatenate all representations
        combined = torch.cat([x, q_out, attn], dim=-1)
        return torch.sigmoid(self.head(combined))
