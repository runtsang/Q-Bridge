import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Quantum‑centric classifier using TorchQuantum ansatz.
    Provides a variational circuit, a quantum kernel, and
    methods for prediction and Gram‑matrix evaluation.
    """

    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 kernel_type: str = "quantum",
                 gamma: float = 1.0,
                 device: str = "cpu"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.q_device = tq.QuantumDevice(n_wires=self.num_qubits, device=device)

        # Build classification ansatz
        self.classifier, self.enc_cls, self.weights_cls, self.obs_cls = self.build_classifier_circuit(num_qubits, depth)

        # Build kernel (only used if kernel_type == 'quantum')
        if kernel_type == "quantum":
            self.kernel, self.enc_k, self.weights_k, self.obs_k = self.build_kernel(num_qubits, depth)

    def build_classifier_circuit(self,
                                 num_qubits: int,
                                 depth: int) -> Tuple[tq.QuantumModule, List[int], List[int], List[tq.QuantumGate]]:
        """
        Returns a TorchQuantum module implementing a variational
        circuit, along with metadata: encoding indices,
        weight indices, and observables.
        """
        class Ansatz(tq.QuantumModule):
            def __init__(self, n_wires: int, depth: int):
                super().__init__()
                self.n_wires = n_wires
                self.depth = depth
                self.encode = tq.ParameterVector("x", n_wires)
                self.var = tq.ParameterVector("theta", n_wires * depth)

            @tq.static_support
            def forward(self, q_device: tq.QuantumDevice) -> None:
                # Data encoding
                for i in range(self.n_wires):
                    q_device.rx(self.encode[i], i)
                # Variational layers
                idx = 0
                for _ in range(self.depth):
                    for i in range(self.n_wires):
                        q_device.ry(self.var[idx], i)
                        idx += 1
                    for i in range(self.n_wires - 1):
                        q_device.cz(i, i + 1)

        ansatz = Ansatz(num_qubits, depth)
        encoding = list(range(num_qubits))
        weight_idx = list(range(num_qubits * depth))
        observables = [tq.z_gate(wire=i) for i in range(num_qubits)]
        return ansatz, encoding, weight_idx, observables

    def build_kernel(self,
                     num_qubits: int,
                     depth: int) -> Tuple[tq.QuantumModule, List[int], List[int], List[tq.QuantumGate]]:
        """
        Quantum kernel built from a data‑encoding ansatz followed
        by a reverse‑encoding of a second sample.
        """
        class KernalAnsatz(tq.QuantumModule):
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
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.ansatz = KernalAnsatz(
                    [
                        {"input_idx": [0], "func": "ry", "wires": [0]},
                        {"input_idx": [1], "func": "ry", "wires": [1]},
                        {"input_idx": [2], "func": "ry", "wires": [2]},
                        {"input_idx": [3], "func": "ry", "wires": [3]},
                    ]
                )

            @tq.static_support
            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x = x.reshape(1, -1)
                y = y.reshape(1, -1)
                self.ansatz(q_device, x, y)
                return torch.abs(q_device.states.view(-1)[0])

        kernel = Kernel(num_qubits)
        encoding = list(range(num_qubits))
        weight_idx = []
        observables = [tq.z_gate(wire=i) for i in range(num_qubits)]
        return kernel, encoding, weight_idx, observables

    def kernel_matrix(self,
                      a: Iterable[torch.Tensor],
                      b: Iterable[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the quantum kernel."""
        kernel = self.kernel
        return np.array([[kernel(self.q_device, x, y).item() for y in b] for x in a])

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate the classifier ansatz and return expectation values."""
        self.classifier(self.q_device)
        results = []
        for i in range(self.num_qubits):
            exp = self.q_device.expectation(tq.z_gate, i)
            results.append(exp)
        return torch.stack(results, dim=0)
