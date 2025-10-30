import torch
import torchquantum as tq
import qiskit
import numpy as np
from typing import Sequence

from torchquantum.functional import func_name_dict

class QuantumKernelAnsatz(tq.QuantumModule):
    """Programmable ansatz for a quantum kernel."""
    def __init__(self, circuit_config: list):
        super().__init__()
        self.circuit_config = circuit_config

    @tq.static_support
    def forward(self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for gate in self.circuit_config:
            params = x[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)
        for gate in reversed(self.circuit_config):
            params = -y[:, gate["input_idx"]] if tq.op_name_dict[gate["func"]].num_params else None
            func_name_dict[gate["func"]](q_device, wires=gate["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated by a fixed ansatz."""
    def __init__(self,
                 n_wires: int = 4,
                 circuit_config: list | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if circuit_config is None:
            circuit_config = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.ansatz = QuantumKernelAnsatz(circuit_config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class QuantumAttention:
    """Quantum self‑attention block built with Qiskit."""
    def __init__(self, n_qubits: int = 4, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

class HybridQuantumKernelAttention:
    """Combines quantum kernel and quantum self‑attention."""
    def __init__(self,
                 n_wires: int = 4,
                 n_qubits: int = 4,
                 circuit_config: list | None = None,
                 backend=None):
        self.kernel = QuantumKernel(n_wires, circuit_config)
        self.attention = QuantumAttention(n_qubits, backend)

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        x = torch.stack(a)
        y = torch.stack(b)
        return self.kernel(x, y).detach().cpu().numpy()

    def attention_counts(self,
                         rotation_params: np.ndarray,
                         entangle_params: np.ndarray,
                         shots: int = 1024) -> dict:
        return self.attention.run(rotation_params, entangle_params, shots)

    def hybrid_counts(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      rotation_params: np.ndarray,
                      entangle_params: np.ndarray,
                      shots: int = 1024) -> dict:
        kernel_counts = self.kernel_matrix(a, b)
        attn_counts = self.attention_counts(rotation_params, entangle_params, shots)
        # Simple fusion: multiply counts by kernel values
        fused = {k: kernel_counts.get(k, 0) * v for k, v in attn_counts.items()}
        return fused

# Back‑compatibility aliases
KernalAnsatz = QuantumKernelAnsatz
Kernel = QuantumKernel
__all__ = [
    "QuantumKernelAnsatz",
    "QuantumKernel",
    "QuantumAttention",
    "HybridQuantumKernelAttention",
    "KernalAnsatz",
    "Kernel",
]
