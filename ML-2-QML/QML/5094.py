import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuanvolutionHybridQ(tq.QuantumModule):
    """
    Quantum counterpart of :class:`QuanvolutionHybrid` that
    projects the high‑dimensional image vector onto a small
    qubit register, applies a random variational layer, and
    returns a measurement‑based feature vector.
    """
    def __init__(self, in_dim: int = 784, n_wires: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_wires = n_wires
        # Linear projection to match the number of qubits
        self.proj = nn.Linear(in_dim, n_wires, bias=False)
        # Encode each projected dimension onto a qubit via ry
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        # Random variational layer to mix information
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        # Measure all qubits in the Pauli‑Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input image flattened to shape [batch, in_dim].

        Returns
        -------
        torch.Tensor
            Measurement result with shape [batch, 2**n_wires].
        """
        bsz = x.shape[0]
        device = x.device
        projected = self.proj(x)  # [B, n_wires]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        self.encoder(qdev, projected)
        self.q_layer(qdev)
        return self.measure(qdev).view(bsz, -1)

__all__ = ["QuanvolutionHybridQ"]
