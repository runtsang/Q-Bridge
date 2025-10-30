"""Hybrid quantum kernel combining quanvolution feature extraction and quantum kernel evaluation."""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two-qubit quantum kernel to 2×2 image patches."""
    def __init__(self):
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
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
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

class QuantumKernelQuanvolution(tq.QuantumModule):
    """Hybrid quantum kernel: quanvolution feature extraction + per‑patch quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.kernel = Kernel()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return Gram matrix between two batches of images."""
        fx = self.quanvolution(x)  # (n, 4*14*14)
        fy = self.quanvolution(y)
        fx = fx.view(x.shape[0], 14 * 14, 4)  # (n, patches, 4)
        fy = fy.view(y.shape[0], 14 * 14, 4)  # (m, patches, 4)
        n = fx.shape[0]
        m = fy.shape[0]
        result = torch.zeros((n, m), device=fx.device)
        for i in range(n):
            for j in range(m):
                # sum quantum kernel over all patches
                patch_kernels = torch.stack(
                    [self.kernel(fx[i:i+1, p], fy[j:j+1, p]) for p in range(14 * 14)],
                    dim=0,
                )
                result[i, j] = patch_kernels.sum()
        return result

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Convenience wrapper returning a NumPy array."""
        return self.forward(a, b).detach().cpu().numpy()

__all__ = ["QuanvolutionFilter", "Kernel", "QuantumKernelQuanvolution"]
