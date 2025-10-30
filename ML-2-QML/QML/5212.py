import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class FraudDetectionHybrid(tq.QuantumModule):
    """
    Quantum kernel module for fraud‑detection.
    Encodes two‑dimensional input vectors into a 4‑qubit circuit,
    applies a fixed ansatz, and returns the absolute overlap
    between two encoded states.
    """
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [0], "func": "ry", "wires": [2]},
                {"input_idx": [1], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.ansatz.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel similarity between two input vectors.
        """
        self.forward(self.q_device, x, y)
        # Return the absolute value of the first amplitude as a similarity score
        return torch.abs(self.q_device.states.view(-1)[0])

__all__ = ["FraudDetectionHybrid"]
