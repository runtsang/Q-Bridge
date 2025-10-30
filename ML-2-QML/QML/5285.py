import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import qiskit
from qiskit import Aer

# ---------- Quantum self‑attention (Qiskit) ----------
class QuantumSelfAttention:
    """Qiskit implementation of a self‑attention‑style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumCircuit(n_qubits)
        self.cr = QuantumCircuit(n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(range(self.n_qubits), range(self.n_qubits))
        return circuit

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# ---------- Quantum kernel ansatz ----------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two feature vectors into a quantum state."""
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
    """Quantum kernel evaluated via a fixed ansatz."""
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

# ---------- Quantum classifier factory ----------
def build_classifier_circuit(num_qubits: int, depth: int):
    """Variational circuit with explicit encoding and layers."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

# ---------- Hybrid quantum‑kernel‑classifier ----------
class QuantumHybridKernelClassifier(tq.QuantumModule):
    """
    Quantum counterpart: a self‑attention block (Qiskit) feeds into a
    TorchQuantum kernel ansatz, followed by a variational classifier.
    """
    def __init__(self, num_qubits: int, depth: int, attention_n_qubits: int = 4):
        super().__init__()
        self.n_qubits = num_qubits
        self.q_device = tq.QuantumDevice(n_wires=num_qubits)
        self.attention = QuantumSelfAttention(attention_n_qubits)
        self.kernel = Kernel()
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel value."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.kernel(x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def classifier_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the variational classifier on input data."""
        self.q_device.reset_states(x.shape[0])
        self.q_device(x, self.encoding, self.weights)
        out = self.measure(self.q_device)
        return out

__all__ = ["QuantumSelfAttention", "KernalAnsatz", "Kernel", "build_classifier_circuit", "QuantumHybridKernelClassifier"]
