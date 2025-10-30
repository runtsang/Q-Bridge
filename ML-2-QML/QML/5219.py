import torch
import torchquantum as tq
import numpy as np
from typing import Sequence

class KernalAnsatz(tq.QuantumModule):
    """Quantum RBF‑style ansatz that encodes data via a list of gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel module that evaluates the overlap between two encoded states."""
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class SamplerQNNQuantum(tq.QuantumModule):
    """Simple quantum sampler that produces a two‑class probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Variational parameters
        self.params = tq.Parameter(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(inputs.shape[0])
        # Encode data
        for i in range(self.n_wires):
            tq.ry(self.q_device, wires=[i], params=inputs[:, i])
        # Variational layer
        tq.cx(self.q_device, wires=[0, 1])
        for i in range(self.n_wires):
            tq.ry(self.q_device, wires=[i], params=self.params[i])
        # Measure probabilities of |00> and |01>
        probs = self.q_device.states.view(-1).abs() ** 2
        p00 = probs[0]
        p01 = probs[1]
        return torch.stack([p00, p01], dim=-1)

class HybridKernelMethod(tq.QuantumModule):
    """
    Quantum‑centric hybrid kernel method that unifies a quantum RBF kernel with
    a quantum sampler head.  The API mirrors the classical counterpart.
    """
    def __init__(self, use_sampler: bool = False) -> None:
        super().__init__()
        self.kernel = Kernel()
        self.sampler = SamplerQNNQuantum() if use_sampler else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the quantum kernel value between two feature vectors."""
        return self.kernel(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return the Gram matrix between two collections of feature vectors."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce a two‑class probability vector using the quantum sampler."""
        if self.sampler is None:
            raise RuntimeError("Sampler not enabled for this instance.")
        return self.sampler(inputs)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "SamplerQNNQuantum",
    "HybridKernelMethod",
]
