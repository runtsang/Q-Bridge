import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that encodes two classical vectors."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
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


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a two‑qubit kernel to 2×2 patches."""
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


class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and the quantum kernel."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, kernel: Kernel, support: torch.Tensor):
        ctx.kernel = kernel
        ctx.support = support
        # Compute kernel features for each support vector
        feats = []
        for sv in support:
            feat = kernel(inputs.unsqueeze(0), sv.unsqueeze(0))
            feats.append(feat)
        result = torch.stack(feats, dim=1)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, result = ctx.saved_tensors
        # Finite‑difference approximation of gradients w.r.t. inputs
        eps = 1e-4
        grads = []
        for i in range(inputs.shape[0]):
            inp_plus = inputs.clone()
            inp_minus = inputs.clone()
            inp_plus[i] += eps
            inp_minus[i] -= eps
            out_plus = ctx.kernel(inp_plus.unsqueeze(0), ctx.support.unsqueeze(0))
            out_minus = ctx.kernel(inp_minus.unsqueeze(0), ctx.support.unsqueeze(0))
            grads.append((out_plus - out_minus) / (2 * eps))
        grads = torch.stack(grads, dim=0)
        return grads * grad_output, None, None


class HybridFCLQuantum(nn.Module):
    """
    Quantum‑augmented fully‑connected layer that chains a quanvolution filter,
    a quantum kernel feature map, and a dense head.
    """
    def __init__(self,
                 in_features: int = 1,
                 n_support: int = 10) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.kernel = Kernel()
        self.support = nn.Parameter(torch.randn(n_support, in_features))
        self.fc = nn.Linear(n_support, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution
        if x.dim() == 4:
            x = self.quanvolution(x)
        # 2. Quantum kernel feature map
        feats = HybridFunction.apply(x, self.kernel, self.support)
        # 3. Dense head
        return self.fc(feats)

    def run(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Legacy ``run`` interface: accepts a 1‑D array of angles and
        returns the mean activation after a tanh non‑linearity.
        """
        values = thetas.view(-1, 1).float()
        activations = torch.tanh(self.fc(values))
        return activations.mean(dim=0)


def FCL() -> HybridFCLQuantum:
    """Factory that matches the legacy API for the quantum version."""
    return HybridFCLQuantum()


__all__ = ["KernalAnsatz", "Kernel", "QuanvolutionFilter",
           "HybridFunction", "HybridFCLQuantum", "FCL"]
