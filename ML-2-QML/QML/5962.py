"""Quantum hybrid classifier with variational kernel and quantum expectation head.

The module implements:
- QuantumVariationalAnsatz: TorchQuantum ansatz for feature map.
- QuantumKernel: compute kernel value using ansatz.
- CNNBackbone: identical to classical counterpart.
- QuantumHybridHead: uses Qiskit circuit to compute expectation value.
- HybridClassifier: end‑to‑end quantum hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class QuantumVariationalAnsatz(tq.QuantumModule):
    """Programmable ansatz that encodes classical data."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.func_list = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernel(tq.QuantumModule):
    """Kernel using the variational ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumVariationalAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class CNNBackbone(nn.Module):
    """Convolutional backbone identical to classical counterpart."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QuantumHybridHead(nn.Module):
    """Hybrid head that computes expectation via Qiskit circuit."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.backend = AerSimulator()
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.ry(qiskit.circuit.Parameter("theta"), range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[0]: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()

        def expectation(count_dict):
            counts_arr = np.array(list(count_dict.values()))
            states = np.array([int(b[0]) for b in count_dict.keys()])  # first qubit
            probs = counts_arr / self.shots
            return np.sum(((-1) ** states) * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(counts)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 1)
        thetas = inputs.squeeze().detach().cpu().numpy()
        exps = self.run(thetas)
        exps = torch.tensor(exps, device=inputs.device, dtype=torch.float32)
        probs = torch.sigmoid(exps)
        return torch.cat((probs, 1 - probs), dim=-1)


class HybridClassifier(nn.Module):
    """End‑to‑end quantum hybrid classifier."""
    def __init__(self, use_kernel: bool = True, n_qubits: int = 2, shots: int = 1024):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = QuantumHybridHead(n_qubits=n_qubits, shots=shots)
        self.use_kernel = use_kernel
        self.kernel = QuantumKernel(n_wires=4) if use_kernel else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        if self.kernel is None:
            raise RuntimeError("Kernel not initialized.")
        return np.array([[self.kernel(a_i, b_j).item() for b_j in b] for a_i in a])
