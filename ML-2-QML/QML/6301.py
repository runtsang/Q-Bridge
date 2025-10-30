import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class HybridAnsatz(tq.QuantumModule):
    """Encodes classical data via a list of parameterised gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # encode x
        for info in self.func_list:
            if info["func"] == "cx":
                func_name_dict[info["func"]](q_device, wires=info["wires"])
            else:
                params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # encode y with negative parameters
        for info in reversed(self.func_list):
            if info["func"] == "cx":
                func_name_dict[info["func"]](q_device, wires=info["wires"])
            else:
                params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class HybridKernel(tq.QuantumModule):
    """
    Quantum kernel that encodes two data points with a parameter‑encoding ansatz
    and returns the overlap of the resulting states.  The ansatz is a
    configurable list of single‑qubit rotations followed by entangling gates.
    """
    def __init__(self, n_wires: int = 4, ansatz=None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if ansatz is None:
            ansatz = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
                {"input_idx": None, "func": "cx", "wires": [0, 1]},
                {"input_idx": None, "func": "cx", "wires": [1, 2]},
                {"input_idx": None, "func": "cx", "wires": [2, 3]},
            ]
        self.ansatz = HybridAnsatz(ansatz)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        a = a.reshape(-1, a.shape[-1]) if a.ndim == 2 else a
        b = b.reshape(-1, b.shape[-1]) if b.ndim == 2 else b
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

def EstimatorQNN():
    """Return a Qiskit EstimatorQNN that mirrors the classical feed‑forward network."""
    from qiskit.circuit import Parameter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import StatevectorEstimator as Estimator

    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)

    observable = SparsePauliOp.from_list([("Y", 1)])

    estimator = Estimator()
    return EstimatorQNN(circuit=qc,
                        observables=observable,
                        input_params=[params[0]],
                        weight_params=[params[1]],
                        estimator=estimator)

__all__ = ["HybridKernel", "EstimatorQNN"]
