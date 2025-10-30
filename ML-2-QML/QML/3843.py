import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """
    Quantum kernel ansatz encoding two classical inputs.
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
    Quantum kernel implemented via TorchQuantum.
    """
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

def kernel_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix between two lists of feature vectors
    using the quantum kernel defined above.
    """
    kernel = Kernel()
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])

class SamplerQNNHybrid(nn.Module):
    """
    Hybrid sampler that exposes both a classical neural sampler and a quantum sampler.
    The quantum sampler uses a simple two‑qubit variational circuit.
    """
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        # Classical sampler network
        self.classical_sampler = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
        # Quantum sampler circuit
        self.quantum_circuit = self._create_quantum_circuit()
        self.sampler = StatevectorSampler()
        # Quantum kernel
        self.kernel = Kernel()
        self.gamma = gamma

    def _create_quantum_circuit(self) -> QuantumCircuit:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Classical forward pass.
        """
        return F.softmax(self.classical_sampler(inputs), dim=-1)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum sampler on the provided classical inputs.
        """
        # bind parameters
        bound = {f"input{i}": val.item() for i, val in enumerate(inputs)}
        bound.update({f"weight{i}": 0.0 for i in range(4)})  # weights fixed to zero
        circ = self.quantum_circuit.bind_parameters(bound)
        result = self.sampler.run(circ).result()
        return result.get_counts()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix using the quantum kernel implementation.
        """
        return kernel_matrix(a, b)

__all__ = ["SamplerQNNHybrid"]
