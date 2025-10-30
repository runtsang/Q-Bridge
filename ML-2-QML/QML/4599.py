from __future__ import annotations

import torch
import torchquantum as tq
import numpy as np
from torchquantum.functional import func_name_dict, op_name_dict

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter applying a random two‑qubit circuit to 2×2 image patches."""
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class KernalAnsatz(tq.QuantumModule):
    """Programmable ansatz for the quantum kernel."""
    def __init__(self, func_list: list) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum RBF‑kernel evaluated via a fixed ansatz."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumCircuit(tq.QuantumModule):
    """Simple two‑parameter circuit for the hybrid expectation head."""
    def __init__(self, n_qubits: int, shift: float) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.q_device = tq.QuantumDevice(n_qubits)
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        self.encoder(self.q_device, thetas)
        measurement = self.measure(self.q_device)
        return measurement[:, 0].unsqueeze(-1)

class Hybrid(tq.QuantumModule):
    """Hybrid layer forwarding activations through a quantum expectation."""
    def __init__(self, n_qubits: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantum_circuit.run(inputs)

class QuanvolutionHybrid(tq.QuantumModule):
    """Full quantum quanvolution pipeline mirroring the classical version."""
    def __init__(self, in_channels: int = 1, n_kernel_wires: int = 4, n_hybrid_wires: int = 1, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(in_channels)
        self.kernel = Kernel(n_wires=n_kernel_wires)
        self.hybrid = Hybrid(n_qubits=n_hybrid_wires, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        prototype = torch.randn(1, feats.shape[1], device=x.device)
        kernel_feats = self.kernel(feats, prototype)
        logits = self.hybrid(kernel_feats)
        probs = torch.cat([logits, 1 - logits], dim=-1)
        return torch.log_softmax(probs, dim=-1)

__all__ = ["QuanvolutionHybrid", "QuanvolutionFilter", "Kernel", "Hybrid", "QuantumCircuit"]
