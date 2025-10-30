from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
import torch
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class SamplerQNNGen177:
    """
    Quantum hybrid sampler.
    Builds a parameterized circuit and a quantum kernel via TorchQuantum.
    """
    def __init__(self, input_dim: int = 2, weight_dim: int = 4) -> None:
        self.input_params = ParameterVector("input", input_dim)
        self.weight_params = ParameterVector("weight", weight_dim)

        self.qc = QuantumCircuit(input_dim)
        for i in range(input_dim):
            self.qc.ry(self.input_params[i], i)
        self.qc.cx(0, 1)

        for i in range(weight_dim):
            self.qc.ry(self.weight_params[i], i % input_dim)

        # Visualise circuit for debugging (optional)
        self.qc.draw("mpl", style="clifford")

        self.sampler = StatevectorSampler()
        self.sampler_qnn = SamplerQNN(
            circuit=self.qc,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler
        )

        # quantum kernel
        self.kernel = self._build_qkernel()

    def _build_qkernel(self) -> tq.QuantumModule:
        class KernalAnsatz(tq.QuantumModule):
            def __init__(self, func_list):
                super().__init__()
                self.func_list = func_list

            @tq.static_support
            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
                # Encode input data
                q_device.reset_states(x.shape[0])
                for info in self.func_list:
                    params = x[:, info["input_idx"][0]] if tq.op_name_dict[info["func"]].num_params else None
                    func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
                # Encode negative of target data
                for info in reversed(self.func_list):
                    params = -y[:, info["input_idx"][0]] if tq.op_name_dict[info["func"]].num_params else None
                    func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        class Kernel(tq.QuantumModule):
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

        return Kernel()

    def sample(self, inputs: torch.Tensor, weights: torch.Tensor) -> np.ndarray:
        """
        Execute the sampler circuit with given parameters.
        """
        param_dict = {p: float(v) for p, v in zip(self.input_params, inputs.squeeze())}
        param_dict.update({p: float(v) for p, v in zip(self.weight_params, weights.squeeze())})
        self.sampler_qnn.set_parameters(param_dict)
        return self.sampler_qnn.sample()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute quantum kernel Gram matrix.
        """
        kernel = self.kernel
        return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["SamplerQNNGen177"]
