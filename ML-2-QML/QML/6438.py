import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

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

class QuanvolutionKernelFilter(tq.QuantumModule):
    """Hybrid quanvolutional filter that computes quantum kernel between 2x2 patches and learned prototypes."""
    def __init__(self, gamma: float = 1.0, num_prototypes: int = 10) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_prototypes = num_prototypes
        # Fixed prototypes in classical space
        self.prototypes = torch.randn(num_prototypes, 4)
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

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel between two batches of vectors."""
        self.ansatz(self.q_device, x, y)
        # amplitude of |0...0> for each batch element
        amp = self.q_device.states.view(x.shape[0], -1)[:, 0]
        return torch.abs(amp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r+2, c:c+2].reshape(bsz, -1)
                patches.append(patch)
        patches = torch.cat(patches, dim=1)  # (bsz, 14*14, 4)
        features_list = []
        for prot in self.prototypes:
            prot_expanded = prot.expand(patches.shape[0] * patches.shape[1], 4)
            kernel_vals = self.kernel(patches.reshape(-1, 4), prot_expanded)
            kernel_vals = kernel_vals.view(patches.shape[0], patches.shape[1])
            features_list.append(kernel_vals)
        features = torch.stack(features_list, dim=2)  # (bsz, 14*14, num_prototypes)
        features = features.permute(0, 2, 1).contiguous()  # (bsz, num_prototypes, 14*14)
        return features.view(features.size(0), -1)

class QuanvolutionKernelClassifier(nn.Module):
    """Classifier that uses the QuanvolutionKernelFilter followed by a linear head."""
    def __init__(self, gamma: float = 1.0, num_prototypes: int = 10, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionKernelFilter(gamma, num_prototypes)
        self.linear = nn.Linear(num_prototypes * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionKernelFilter", "QuanvolutionKernelClassifier"]
