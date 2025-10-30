"""ConvGen025: quantum implementation of the hybrid conv‑reg‑kernel module.

The class replaces each classical component with a quantum‑enabled counterpart:
* QuanvCircuit – a variational ansatz that emulates the classical convolution filter.
* QuantumRegressionModel – a TorchQuantum module that encodes the input state,
  applies a random layer and trainable rotations, and maps the expectation values
  to a scalar output.
* QuantumKernel – a fixed TorchQuantum ansatz that evaluates a quantum kernel
  between two classical vectors.
* QuantumClassifier – a Qiskit circuit that performs data‑encoding, a shallow
  variational depth, and measures Z‑observables for a binary decision.
All components expose a unified API compatible with the classical interface.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Sequence
import qiskit
from qiskit import Aer, QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
import torchquantum as tq

# Quantum filter
class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        # Encoding: rotate each qubit by π if the pixel exceeds the threshold
        for q in range(self.n_qubits):
            self.circuit.rx(np.pi, q)
        self.circuit.barrier()
        # Add a shallow random circuit
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in data:
            bind = {f"theta{i}": (np.pi if val > self.threshold else 0) for i, val in enumerate(row)}
            param_binds.append(bind)
        job = execute(self.circuit,
                       self.backend,
                       shots=self.shots,
                       parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        total_ones = sum(sum(int(bit) for bit in key) * cnt for key, cnt in result.items())
        return total_ones / (self.shots * self.n_qubits)

# Quantum regression model
class QuantumRegressionModel(nn.Module):
    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.random_layer = tq.RandomLayer(n_ops=30, wires=range(num_wires))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)
        feats = self.measure(qdev)
        return self.head(feats).squeeze(-1)

# Quantum kernel
class QuantumKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.qdev.reset_states(x.shape[0])
        for i in range(self.n_wires):
            tq.RY(x[:, i], wires=i)(self.qdev)
        for i in range(self.n_wires):
            tq.RY(-y[:, i], wires=i)(self.qdev)
        return torch.abs(self.qdev.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Quantum classifier
class QuantumClassifier:
    def __init__(self, num_qubits: int, depth: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)
        self.circuit = QuantumCircuit(num_qubits)
        # Data encoding
        for q in range(num_qubits):
            self.circuit.rx(self.encoding[q], q)
        # Variational depth
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                self.circuit.ry(self.weights[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                self.circuit.cz(q, q + 1)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")

    def run(self, data: np.ndarray) -> torch.Tensor:
        data = data.reshape(1, self.num_qubits)
        param_binds = []
        for row in data:
            bind = {f"x{i}": row[i] for i in range(self.num_qubits)}
            for idx, val in enumerate(row):
                bind[f"theta{idx}"] = val
            param_binds.append(bind)
        job = execute(self.circuit,
                       self.backend,
                       shots=1024,
                       parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        probs = {k: v / 1024 for k, v in result.items()}
        expectations = torch.zeros(self.num_qubits)
        for bitstring, p in probs.items():
            for i, bit in enumerate(reversed(bitstring)):
                expectations[i] += (1 if bit == '1' else -1) * p
        return expectations

# Unified class
class ConvGen025:
    """Quantum implementation mirroring the classical ConvGen025."""
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.5,
                 num_wires: int = 4,
                 num_qubits: int = 4,
                 depth: int = 2):
        backend = Aer.get_backend("qasm_simulator")
        self.filter = QuanvCircuit(kernel_size, backend, threshold=threshold)
        self.regressor = QuantumRegressionModel(num_wires)
        self.kernel = QuantumKernel()
        self.classifier = QuantumClassifier(num_qubits, depth)

    def filter_data(self, data: np.ndarray) -> float:
        return self.filter.run(data)

    def regress(self, x: np.ndarray) -> float:
        x_t = torch.tensor(x, dtype=torch.cfloat)
        return self.regressor(x_t).item()

    def compute_kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_t = torch.tensor(a, dtype=torch.cfloat)
        b_t = torch.tensor(b, dtype=torch.cfloat)
        return kernel_matrix(a_t, b_t)

    def classify(self, data: np.ndarray) -> torch.Tensor:
        return self.classifier.run(data)

__all__ = ["ConvGen025"]
