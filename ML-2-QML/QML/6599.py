import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumHead(tq.QuantumModule):
    """Hybrid quantum head that maps a 4‑dimensional classical feature vector to a scalar via a 4‑qubit variational circuit."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Random layers for expressivity
        self.random_layers = nn.ModuleList(
            [tq.RandomLayer(n_ops=10, wires=list(range(n_wires))) for _ in range(2)]
        )
        # Trainable single‑qubit rotations
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Measurement of all qubits
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_wires). Each element is used as a rotation angle.
        Returns
        -------
        torch.Tensor
            Shape (batch, 1) – expectation value of the first qubit.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode each feature as a rotation around Y
        for i in range(self.n_wires):
            tqf.ry(qdev, x[:, i], wires=i)

        # Apply random layers
        for rl in self.random_layers:
            rl(qdev)

        # Apply trainable rotations
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)

        # Measure all qubits
        out = self.measure(qdev)  # shape (batch, n_wires)
        # Return expectation of the first qubit as a scalar
        return out[:, 0].unsqueeze(-1)

__all__ = ["QuantumHead"]
