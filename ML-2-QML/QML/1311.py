import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernelMethod(tq.QuantumModule):
    """Hybrid quantum kernel with a parameterized variational circuit."""
    def __init__(self, n_wires: int = 4, depth: int = 2, encoding: str = 'ry', variational: bool = True):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.encoding = encoding
        self.variational = variational
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        ops = []
        # Encoding layer
        for i in range(self.n_wires):
            ops.append({"input_idx": [i], "func": self.encoding, "wires": [i]})
        # Variational layers
        if self.variational:
            for _ in range(self.depth):
                for i in range(self.n_wires):
                    ops.append({"input_idx": [i], "func": "ry", "wires": [i]})
                for i in range(self.n_wires - 1):
                    ops.append({"input_idx": [], "func": "cx", "wires": [i, i + 1]})
        return ops

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.ansatz:
            if info["func"] == "cx":
                func_name_dict[info["func"]](q_device, wires=info["wires"])
            else:
                params = x[:, info["input_idx"]]
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Encode y with negative sign
        for info in reversed(self.ansatz):
            if info["func"] == "cx":
                func_name_dict[info["func"]](q_device, wires=info["wires"])
            else:
                params = -y[:, info["input_idx"]]
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix between two sets of data."""
        n, m = a.shape[0], b.shape[0]
        K = torch.empty((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = self.kernel_value(a[i:i+1], b[j:j+1])
        return K
