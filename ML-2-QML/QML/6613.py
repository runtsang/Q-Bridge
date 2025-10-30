import torch
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
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

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
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

class SamplerQNN:
    """
    Quantum sampler that combines a Qiskit variational circuit with a TorchQuantum kernel.
    The sampler returns probability distributions over 2 qubits and can evaluate a quantum
    kernel for feature mapping.
    """
    def __init__(self):
        # Define input and weight parameters for the variational circuit
        self.inputs2 = ParameterVector("input", 2)
        self.weights2 = ParameterVector("weight", 4)

        # Build the variational circuit
        self.qc2 = QuantumCircuit(2)
        self.qc2.ry(self.inputs2[0], 0)
        self.qc2.ry(self.inputs2[1], 1)
        self.qc2.cx(0, 1)
        self.qc2.ry(self.weights2[0], 0)
        self.qc2.ry(self.weights2[1], 1)
        self.qc2.cx(0, 1)
        self.qc2.ry(self.weights2[2], 0)
        self.qc2.ry(self.weights2[3], 1)

        # Sampler primitive
        self.sampler = Sampler()
        # Wrap into Qiskit Machine Learning SamplerQNN
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=self.qc2,
            input_params=self.inputs2,
            weight_params=self.weights2,
            sampler=self.sampler
        )

        # Quantum kernel
        self.kernel = Kernel()

    def sample(self, input_vals: np.ndarray, weight_vals: np.ndarray) -> np.ndarray:
        """
        Execute the variational circuit with specified parameters and return measurement probabilities.
        :param input_vals: array of shape (2,)
        :param weight_vals: array of shape (4,)
        :return: probability distribution over 2 qubits.
        """
        param_dict = dict(zip(self.inputs2.params, input_vals))
        param_dict.update(dict(zip(self.weights2.params, weight_vals)))
        result = self.sampler_qnn.run(param_dict)
        return result.probabilities()

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two classical data points.
        :param x: Tensor of shape (batch, 4)
        :param y: Tensor of shape (batch, 4)
        :return: Tensor of kernel values.
        """
        return self.kernel(x, y)

__all__ = ["SamplerQNN", "Kernel", "KernalAnsatz"]
