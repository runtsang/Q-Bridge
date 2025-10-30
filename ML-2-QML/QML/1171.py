import pennylane as qml
import torch
import torch.nn as nn

class QuantumNATEnhanced(nn.Module):
    """
    Quantum model that encodes images into a 4‑qubit variational circuit.
    The image is first compressed to a 4×4 parameter vector via a lightweight CNN,
    then reshaped into parameters for a hardware‑efficient ansatz.
    The circuit outputs expectation values of PauliZ on each qubit,
    which are linearly transformed to the final 4 outputs.
    """
    def __init__(self, device="default.qubit", wires=4):
        super().__init__()
        self.wires = wires
        self.device = qml.device(device, wires=wires)
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        # Parameter layer to produce 4 parameters per qubit
        self.param_layer = nn.Linear(4*7*7, wires*4)
        # Readout linear layer
        self.readout = nn.Linear(wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def _quantum_circuit(self, params):
        # params shape: (batch, wires*4)
        for i in range(self.wires):
            qml.RY(params[..., i*4 + 0], wires=i)
            qml.RZ(params[..., i*4 + 1], wires=i)
            qml.RX(params[..., i*4 + 2], wires=i)
            qml.CNOT(wires=[i, (i+1)%self.wires])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        enc = self.encoder(x)
        flat = enc.view(bsz, -1)
        params = self.param_layer(flat)
        # Quantum circuit
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(params_batch):
            return self._quantum_circuit(params_batch)
        out = circuit(params)  # shape (bsz, wires)
        out = self.readout(out)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
