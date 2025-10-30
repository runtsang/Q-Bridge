import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Quantum data‑encoding ansatz used by the kernel."""
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

class SelfAttentionHybrid:
    """Quantum‑enhanced self‑attention that combines a Qiskit circuit with a TorchQuantum kernel."""
    def __init__(self, n_qubits: int = 4, gamma: float = 1.0) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.kernel = Kernel()

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       inputs: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            # Encode raw input as a Ry rotation
            circuit.ry(inputs[i], i)
            # Additional parameterised rotations
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        # Classical‑style projection
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.n_qubits, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.n_qubits, -1), dtype=torch.float32)

        # Quantum‑kernel similarity between query and key
        kernel_mat = np.array([[self.kernel(q.unsqueeze(0), k.unsqueeze(0)).item() for k in key] for q in query])
        kernel_mat = kernel_mat / kernel_mat.sum(axis=1, keepdims=True)

        # Weighted sum of values
        output = kernel_mat @ inputs

        # Execute the Qiskit circuit for entanglement and measurement
        circuit = self._build_circuit(rotation_params, entangle_params, inputs)
        job = execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) / shots
                          for i in range(2 ** self.n_qubits)])

        # Simple fusion: modulate output by the first |output| probabilities
        combined = output * probs[:output.shape[0]] if probs.size >= output.shape[0] else output
        return combined

def SelfAttention() -> SelfAttentionHybrid:
    """Convenience factory matching the original API."""
    return SelfAttentionHybrid()

__all__ = ["SelfAttention", "SelfAttentionHybrid", "Kernel", "KernalAnsatz"]
