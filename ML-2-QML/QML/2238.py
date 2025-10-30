import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Iterable, Tuple, List

class KernalAnsatz(tq.QuantumModule):
    """
    Quantum kernel ansatz that encodes two classical vectors x and y
    with opposite sign parameters and then measures the overlap.
    """
    def __init__(self, func_list: List[dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device,
                                        wires=info["wires"],
                                        params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device,
                                        wires=info["wires"],
                                        params=params)

class Kernel(tq.QuantumModule):
    """
    Wrapper that exposes the quantum kernel as a callable module.
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

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

class QuantumClassifierModel(tq.QuantumModule):
    """
    Quantum classifier that mirrors the classical architecture but uses
    a variational circuit plus the quantum kernel for feature extraction.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 gamma: float = 1.0):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.gamma = gamma

        # Metadata that matches the classical side
        self.encoding = tq.ParameterVector("x", num_qubits)
        self.weights = tq.ParameterVector("theta", num_qubits * depth)
        self.observables = [tq.SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                            for i in range(num_qubits)]

        self.q_device = tq.QuantumDevice(n_wires=num_qubits)
        self.ansatz = self._build_ansatz()

        # Quantum kernel for feature extraction
        self.kernel = Kernel()

    def _build_ansatz(self) -> tq.QuantumModule:
        circuit = tq.QuantumCircuit(self.num_qubits)
        for param, qubit in zip(self.encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the variational circuit and return expectation values
        of the Z observables – these serve as quantum feature maps.
        """
        self.q_device.reset_states(x.shape[0])
        self.ansatz(self.q_device, x)
        return torch.stack([self.q_device.expectation(op).mean() for op in self.observables],
                           dim=-1)

    def quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two samples using the
        built‑in Kernel module.
        """
        return self.kernel(x, y)

    # ------------------------------------------------------------------
    # Compatibility helpers – expose the same API as the seed
    # ------------------------------------------------------------------
    def build_classifier_circuit(self) -> Tuple[tq.QuantumCircuit,
                                               Iterable,
                                               Iterable,
                                               List[tq.SparsePauliOp]]:
        return self.ansatz.circuit, list(self.encoding), list(self.weights), self.observables

__all__ = ["QuantumClassifierModel", "KernalAnsatz", "Kernel"]
