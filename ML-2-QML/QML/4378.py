import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
from tq.backends import get_backend
from typing import Sequence

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
            tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            tq.func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class FCL(tq.QuantumModule):
    """Parameterized quantum circuit for a fully connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = get_backend("qasm_simulator")
        self.circuit = tq.QuantumCircuit(n_qubits)
        self.theta = tq.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        params = {self.theta: thetas}
        job = tq.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[params])
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class SelfAttention(tq.QuantumModule):
    """Quantum self‑attention block."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> tq.QuantumCircuit:
        circuit = tq.QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure_all()
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = tq.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridKernelModel(tq.QuantumModule):
    """
    Quantum hybrid kernel model that combines a quantum feature extractor
    (QFCModel), a quantum fully connected layer (FCL), and a quantum
    self‑attention block.  The kernel is evaluated by the KernalAnsatz
    ansatz and can be used directly for kernel‑based learning.
    """
    def __init__(self, n_wires: int = 4, use_attention: bool = False):
        super().__init__()
        self.qfc = QFCModel()
        self.fcl = FCL()
        self.use_attention = use_attention
        self.attention = SelfAttention(n_qubits=n_wires) if use_attention else None
        self.kernel = Kernel(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        features_x = self.qfc(x)
        features_y = self.qfc(y)

        # Optional attention
        if self.use_attention:
            rot = np.random.randn(self.kernel.n_wires * 3)
            ent = np.random.randn(self.kernel.n_wires - 1)
            att_x = self.attention.run(rot, ent)
            att_y = self.attention.run(rot, ent)
            # Convert counts to a simple vector (here we just take the sum of counts)
            features_x = torch.tensor([sum(att_x.values())], dtype=torch.float32)
            features_y = torch.tensor([sum(att_y.values())], dtype=torch.float32)

        # Fully connected layer
        thetas = np.linspace(0, np.pi, self.kernel.n_wires)
        fcl_out_x = self.fcl.run(thetas)
        fcl_out_y = self.fcl.run(thetas)

        # Compute kernel
        return self.kernel(features_x, features_y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QFCModel",
    "FCL",
    "SelfAttention",
    "HybridKernelModel",
]
