import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class HybridFullyConnectedLayer(tq.QuantumModule):
    """
    Quantum kernel layer that mirrors the classical counterpart.  It
    encodes each input vector with a Ry‑ansatz, applies a reverse encoding
    for the second vector, and returns the absolute overlap as the kernel
    value.  Parameters are clipped and an affine post‑processing is
    provided (scale & shift) to match the fraud‑detection scaling logic.
    """
    def __init__(self, n_wires: int, clip: bool = True, bound: float = 5.0):
        super().__init__()
        self.n_wires = n_wires
        self.clip = clip
        self.bound = bound
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = self._build_ansatz()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.shift = torch.nn.Parameter(torch.tensor(0.0))

    def _build_ansatz(self):
        return [
            {"input_idx": [i], "func": "ry", "wires": [i]}
            for i in range(self.n_wires)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.clip:
            x = torch.clamp(x, -self.bound, self.bound)
            y = torch.clamp(y, -self.bound, self.bound)
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = x[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = -y[:, info["input_idx"]]
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0]) * self.scale + self.shift

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n_a, n_b = a.shape[0], b.shape[0]
        out = torch.empty(n_a, n_b, device=a.device)
        for i in range(n_a):
            for j in range(n_b):
                out[i, j] = self.kernel(a[i:i+1], b[j:j+1])
        return out

__all__ = ["HybridFullyConnectedLayer"]
