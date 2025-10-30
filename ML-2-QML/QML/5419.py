import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 1. Quantum kernel layer that maps measurement outcomes to RBF similarities
# --------------------------------------------------------------------------- #
class QuantumKernelLayer(tq.QuantumModule):
    """Computes an RBF kernel between quantum measurement vectors and
    trainable support vectors."""
    def __init__(self,
                 n_wires: int,
                 n_support: int = 16,
                 gamma: float = 1.0):
        super().__init__()
        self.n_wires = n_wires
        self.support = nn.Parameter(torch.randn(n_support, n_wires))
        self.gamma = gamma
        # Measurement of all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # qdev.states shape: (bsz, 2**n_wires)
        # Convert to expectation values of Z (already provided by MeasureAll)
        features = self.measure(qdev)                   # → (bsz, n_wires)
        diff = features.unsqueeze(1) - self.support.unsqueeze(0)  # (bsz, n_support, n_wires)
        dist_sq = torch.sum(diff * diff, dim=-1)        # (bsz, n_support)
        return torch.exp(-self.gamma * dist_sq)         # RBF kernel features

# --------------------------------------------------------------------------- #
# 2. Quantum hybrid model: linear pre‑encoder → variational circuit → kernel → head
# --------------------------------------------------------------------------- #
class QuantumNATHybrid(tq.QuantumModule):
    """
    Quantum counterpart of the classical hybrid model.
    An image is linearly projected into n_wires qubits, processed by a
    variational circuit, and the resulting expectation values are fed
    through the same trainable RBF kernel and linear regression head.
    """
    def __init__(self,
                 num_wires: int = 4,
                 n_support: int = 16,
                 gamma: float = 1.0):
        super().__init__()
        self.n_wires = num_wires

        # Classical linear encoder that maps the 28×28 image to n_wires amplitudes
        self.linear_encoder = nn.Linear(28 * 28, num_wires)

        # Variational quantum circuit
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(num_wires)))

        # Kernel mapping on measurement outcomes
        self.kernel_layer = QuantumKernelLayer(num_wires, n_support, gamma)

        # Regression head
        self.head = nn.Linear(n_support, 1)
        self.norm = nn.BatchNorm1d(n_support)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of single‑channel images of shape (bsz, 1, 28, 28).
        """
        bsz = x.shape[0]
        # Flatten image and linearly project to qubit amplitudes
        flat = x.view(bsz, -1)                          # (bsz, 784)
        encoded = self.linear_encoder(flat)             # (bsz, n_wires)

        # Quantum device and circuit
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, encoded)
        self.q_layer(qdev)

        # Apply the kernel layer on measurement outcomes
        kfeat = self.kernel_layer(qdev)                 # (bsz, n_support)
        kfeat = self.norm(kfeat)
        out = self.head(kfeat)                          # (bsz, 1)
        return out.squeeze(-1)                          # (bsz,)

__all__ = ["QuantumNATHybrid"]
