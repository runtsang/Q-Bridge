import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Variational quanvolutional filter.

    A 4‑qubit variational circuit is applied to each 2×2 patch of a
    28×28 image.  Each pixel value is encoded into a Ry rotation
    on a dedicated qubit.  The circuit consists of `depth` layers,
    each containing a layer of CNOT entanglers followed by a
    layer of RZ rotations with trainable parameters.  The
    measurement is a full Pauli‑Z readout, yielding a 4‑dimensional
    feature vector per patch.  The filter is fully batch‑wise
    differentiable and can be trained end‑to‑end with a classical
    head.
    """
    def __init__(self, depth: int = 3):
        super().__init__()
        self.n_wires = 4
        self.depth = depth
        # Feature‑mapping: encode pixel values into Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational ansatz: depth layers of CNOT + RZ
        self.var_circuit = tq.QuantumModule()
        for d in range(depth):
            # CNOT entanglement
            for i in range(self.n_wires - 1):
                self.var_circuit.append(tq.CNOTGate(wires=[i, i + 1]))
            # RZ rotations with trainable params
            for w in range(self.n_wires):
                self.var_circuit.append(tq.RZGate(wires=[w]))
        # Measurement of all qubits in Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.var_circuit(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
