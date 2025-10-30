import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch
import numpy as np

class KernalAnsatz(tq.QuantumModule):
    """
    Quantum feature map that encodes classical data using a list of gates.
    """
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
    """
    Quantum kernel module that evaluates the overlap of two feature-encoded states.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a, b):
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class EstimatorQNNHybrid(tq.QuantumModule):
    """
    Quantum neural network estimator that wraps the quantum kernel and a linear readout.
    """
    def __init__(self, n_wires: int = 4, lambda_reg: float = 1e-3):
        super().__init__()
        self.kernel = Kernel(n_wires)
        self.lambda_reg = lambda_reg
        self.w = None
        self.train_X = None

    def train(self, X: torch.Tensor, y: torch.Tensor) -> None:
        K = self.kernel(X, X).squeeze()
        self.w = torch.linalg.solve(K + self.lambda_reg * torch.eye(K.shape[0]), y)
        self.train_X = X

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.train_X is None or self.w is None:
            raise RuntimeError("Model has not been trained.")
        phi = self.kernel(self.train_X, inputs).squeeze()
        return phi @ self.w
