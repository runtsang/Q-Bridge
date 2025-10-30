import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler

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

class SamplerQNN(tq.QuantumModule):
    """Hybrid quantum sampler with kernel evaluation and parameterized circuit."""
    def __init__(self, n_qubits: int = 2, n_params: int = 4):
        super().__init__()
        # Quantum device for sampling
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        # Parameterized circuit for sampling
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", n_params)
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        for i in range(n_params):
            self.circuit.ry(self.weights[i], i % n_qubits)
        self.circuit.cx(0, 1)
        # Sampler primitive
        self.sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(circuit=self.circuit,
                                       input_params=self.inputs,
                                       weight_params=self.weights,
                                       sampler=self.sampler)
        # Quantum kernel ansatz
        self.kernel_ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return quantum kernel value between two inputs."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.kernel_ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Compute Gram matrix using the quantum kernel."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def sample(self, inputs: list[float], weights: list[float], shots: int = 1024) -> dict:
        """Sample from the parameterized circuit given classical parameters."""
        param_bindings = {**{self.inputs[i]: inputs[i] for i in range(2)},
                          **{self.weights[i]: weights[i] for i in range(len(weights))}}
        result = self.sampler_qnn.sample(param_bindings, shots=shots)
        return result.get_counts(self.circuit)
