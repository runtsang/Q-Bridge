import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from torchquantum.functional import func_name_dict, op_name_dict

class KernalAnsatz(tq.QuantumModule):
    """
    Quantum ansatz that encodes two input vectors into a shared quantum device
    and measures the overlap.  The implementation mirrors the classical
    :class:`KernalAnsatz` but operates on a :class:`QuantumDevice`.
    """
    def __init__(self, func_list: list[dict]) -> None:
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
    """
    Quantum kernel evaluated via a fixed TorchQuantum ansatz.
    """
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

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between two collections of tensors."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridSamplerKernel(tq.QuantumModule):
    """
    Quantum‑augmented sampler that uses a quantum kernel to embed classical
    data, then passes the resulting similarity vector through a small
    variational circuit to produce a two‑class probability distribution.
    """
    def __init__(self, n_wires: int = 2, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.kernel = Kernel()
        self.hidden_dim = hidden_dim
        self.sampler_params = nn.Parameter(torch.randn(2))
        self.sampler = None

    def _build_sampler(self, num_refs: int) -> None:
        hidden = self.hidden_dim or num_refs
        self.sampler = nn.Sequential(
            nn.Linear(num_refs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of input vectors of shape (B, D).
        references : torch.Tensor
            Reference vectors of shape (R, D) used to compute kernel similarities.

        Returns
        -------
        torch.Tensor
            Probabilities of shape (B, 2).
        """
        # Compute kernel similarities
        k_vals = []
        for ref in references:
            k_vals.append(self.kernel(x, ref))
        k_mat = torch.stack(k_vals)  # shape (R,)
        if self.sampler is None:
            self._build_sampler(k_mat.shape[0])

        # Encode the similarity vector as a rotation angle on the first qubit
        theta = torch.sum(k_mat)
        self.q_device.reset_states(1)
        self.q_device.ry(theta, 0)

        # Apply trainable variational parameters
        self.q_device.ry(self.sampler_params[0], 0)
        self.q_device.ry(self.sampler_params[1], 1)
        self.q_device.cx(0, 1)

        # Extract measurement probabilities of qubit‑0 = |0⟩
        probs = torch.abs(self.q_device.states.view(-1)**2)[0:2]
        prob0 = probs[0] + probs[1]  # sum over qubit‑1 states where qubit‑0 is 0
        return torch.tensor([prob0, 1 - prob0])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "HybridSamplerKernel"]
