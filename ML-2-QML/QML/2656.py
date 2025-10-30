import pennylane as qml
import torch
from torch import nn
from typing import Tuple, Optional

class UnifiedAutoencoderTransformer(nn.Module):
    """
    Hybrid autoencoder that combines classical encoding, a transformer encoder,
    and a quantum variational circuit implemented with Pennylane.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 transformer_layers: int = 2,
                 transformer_heads: int = 4,
                 transformer_ffn: int = 128,
                 num_trash: int = 2,
                 quantum_reps: int = 5,
                 q_device: Optional[object] = None):
        super().__init__()
        # Classical autoencoder
        self.enc_lin = nn.Linear(input_dim, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, input_dim)

        # Transformer encoder
        self.transformer = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=latent_dim,
                                         nhead=transformer_heads,
                                         dim_feedforward=transformer_ffn,
                                         dropout=dropout)
              for _ in range(transformer_layers)]
        )

        # Quantum device and ansatz
        self.qdev = qml.device("default.qubit", wires=latent_dim + num_trash)
        self.ansatz = qml.templates.BasicEntanglerLayers(
            wires=range(latent_dim + num_trash), reps=quantum_reps
        )

    def _quantum_forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply the variational circuit to a single latent vector.
        z: (latent_dim,)
        Returns: (latent_dim,)
        """
        @qml.qnode(self.qdev, interface="torch")
        def circuit(z_vec):
            # Pad with zeros for trash qubits
            padded = torch.cat([z_vec, torch.zeros(self.qdev.wires, device=z_vec.device)])
            qml.QubitStateVector(padded, wires=range(self.qdev.wires))
            qml.apply(self.ansatz)
            # Return expectation values of each qubit
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.qdev.wires)], dim=0)

        # The circuit expects a vector of length latent_dim + num_trash
        return circuit(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        z = self.enc_lin(x)
        # Transformer encoder
        z = self.transformer(z.unsqueeze(1)).squeeze(1)
        # Quantum enhancement (batch‑wise)
        z_quantum = []
        for i in range(z.size(0)):
            z_quantum.append(self._quantum_forward(z[i]))
        z = torch.stack(z_quantum, dim=0)
        # Classical decoding
        return self.dec_lin(z)
