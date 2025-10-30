import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelAnsatz(tq.QuantumModule):
    """Programmable quantum kernel that encodes two classical vectors."""
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

class QuantumKernel(tq.QuantumModule):
    """Evaluates the overlap between two encoded states."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
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

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Random two‑qubit kernel applied to each 2×2 patch of an image."""
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

class ConvGenQuantum(torch.nn.Module):
    """
    Quantum‑centric analogue of ConvGen.
    Supports classical, quantum, and hybrid (quanvolution) modes.
    """
    def __init__(self, conv_type: str = "hybrid"):
        super().__init__()
        self.conv_type = conv_type.lower()
        if self.conv_type == "classical":
            self.filter = torch.nn.Conv2d(1, 1, kernel_size=2, bias=True)
        elif self.conv_type == "quantum":
            self.filter = QuantumKernel()
        else:  # hybrid
            self.filter = QuanvolutionFilterQuantum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv_type == "quantum":
            # kernel between each image and a zero vector (acts as a scalar)
            zero = torch.zeros_like(x[:,0].flatten(1))
            return self.filter(x[:,0].flatten(1), zero)
        elif self.conv_type == "classical":
            return self.filter(x)
        else:
            return self.filter(x)

__all__ = ["ConvGenQuantum", "QuantumKernelAnsatz", "QuantumKernel", "QuanvolutionFilterQuantum"]
