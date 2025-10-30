"""Hybrid quantum model combining patchwise quantum kernel mapping with a linear head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes two input vectors into a single quantum state."""
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
    """Quantum kernel computed via a fixed ansatz."""
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

class QuantumRBFKernelFeature(tq.QuantumModule):
    """Quantum RBF kernel feature mapping for each patch using a learned set of prototypes."""
    def __init__(self, num_prototypes: int = 32, gamma: float = 1.0):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 4))
        self.kernel = QuantumKernel()

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        patch: (batch, 4)
        returns: (batch, num_prototypes)
        """
        batch = patch.shape[0]
        features = []
        for i in range(self.num_prototypes):
            proto = self.prototypes[i].unsqueeze(0)  # (1, 4)
            sim = self.kernel(patch, proto)  # (batch, 1)
            features.append(sim)
        return torch.cat(features, dim=1)  # (batch, num_prototypes)

class QuanvolutionHybrid(tq.QuantumModule):
    """Quantum hybrid model: classical convolution -> quantum RBF kernel feature mapping -> linear head."""
    def __init__(self, num_prototypes: int = 32, gamma: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.kernel_feat = QuantumRBFKernelFeature(num_prototypes, gamma)
        self.linear = nn.Linear(num_prototypes * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)  # (batch, 4, 14, 14)
        batch = conv_out.shape[0]
        conv_out = conv_out.view(batch, 4, -1)  # (batch, 4, 196)
        # compute quantum kernel features for each patch
        feats = []
        for i in range(conv_out.shape[-1]):  # 196 patches
            patch = conv_out[:, :, i]  # (batch, 4)
            feat = self.kernel_feat(patch)  # (batch, num_prototypes)
            feats.append(feat)
        feats = torch.stack(feats, dim=2)  # (batch, num_prototypes, 196)
        feats = feats.view(batch, -1)  # (batch, num_prototypes * 196)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
