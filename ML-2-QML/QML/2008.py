"""Quantum Quanvolution using a trainable variational circuit."""
import torch
import torchquantum as tq


class Quanvolution__gen209(tq.QuantumModule):
    """Variational quanvolutional filter that processes 2×2 image patches."""
    def __init__(self, n_wires: int = 4, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers

        # Trainable RY rotation parameters
        self.ry_params = tq.Parameter(torch.randn(1, n_wires, n_layers))
        # Entangling CNOT pattern (linear chain)
        self.cnot_pattern = [(i, i + 1) for i in range(n_wires - 1)]

        # Measurement operator
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, 1, 28, 28) (grayscale images)
        Returns: Tensor of shape (B, 196*4) – flattened measurements
        """
        bsz, _, h, w = x.shape
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        measurements = []

        # Iterate over 2×2 patches with stride 2
        for r in range(0, h, 2):
            for c in range(0, w, 2):
                patch = x[:, 0, r:r+2, c:c+2].reshape(bsz, -1)  # (B,4)
                self.encoder(qdev, patch)
                # Variational layers
                for l in range(self.n_layers):
                    tq.RY(self.ry_params[:, :, l], wires=range(self.n_wires))(qdev)
                    for ctrl, tgt in self.cnot_pattern:
                        tq.CNOT(ctrl, tgt)(qdev)
                measurement = self.measure(qdev)  # (B,4)
                measurements.append(measurement)

        return torch.cat(measurements, dim=1)

    def encoder(self, qdev: tq.QuantumDevice, patch: torch.Tensor):
        """Encode a 2×2 patch into qubits using RY rotations."""
        for i in range(self.n_wires):
            tq.RY(patch[:, i], wires=[i])(qdev)

__all__ = ["Quanvolution__gen209"]
