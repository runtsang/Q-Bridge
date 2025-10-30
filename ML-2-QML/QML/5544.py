import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable, Tuple

class QuanvCircuit:
    """Parameterised quanvolution filter."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class LinearQuantumCircuit:
    """Simple 1‑qubit parameterised circuit for a linear layer."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class KernalAnsatz(tq.QuantumModule):
    """TorchQuantum ansatz for the quantum kernel."""
    def __init__(self, func_list: list[dict]) -> None:
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
    """Quantum kernel implemented with TorchQuantum."""
    def __init__(self) -> None:
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

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class HybridFCL:
    """Hybrid quantum‑classical fully‑connected layer with convolution, kernel, and classifier."""
    def __init__(self,
                 conv_kernel_size: int = 2,
                 linear_qubits: int = 1,
                 depth: int = 2,
                 kernel_qubits: int = 4,
                 num_classes: int = 2,
                 reference_points: int = 10,
                 shots: int = 1024) -> None:
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv = QuanvCircuit(conv_kernel_size, backend, shots, threshold=127)
        self.linear = LinearQuantumCircuit(linear_qubits, backend, shots)
        self.kernel = Kernel()
        self.ref_points = torch.randn(reference_points, 2)  # match feature dimension
        self.classifier, self.encoding, self.weights, self.observables = build_classifier_circuit(reference_points, depth)
        self.backend = backend
        self.shots = shots

    def run(self, data: np.ndarray, thetas: Iterable[float] | None = None) -> np.ndarray:
        conv_out = self.conv.run(data)
        linear_out = self.linear.run([conv_out])  # use conv output as theta
        features = torch.tensor([conv_out, linear_out[0]], dtype=torch.float32).unsqueeze(0)
        kernel_vec = torch.zeros(self.ref_points.shape[0], dtype=torch.float32)
        for i, ref in enumerate(self.ref_points):
            kernel_vec[i] = self.kernel(features, ref.unsqueeze(0))
        encode_params = kernel_vec.squeeze().tolist()
        param_bind = {param: val for param, val in zip(self.encoding, encode_params)}
        if thetas is None:
            thetas = np.random.uniform(0, 2 * np.pi, size=len(self.weights))
        weight_bind = {param: val for param, val in zip(self.weights, thetas)}
        param_binds = [ {**param_bind, **weight_bind} ]
        job = qiskit.execute(
            self.classifier,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds
        )
        result = job.result()
        counts = result.get_counts(self.classifier)
        probs = np.zeros(len(self.observables))
        for key, val in counts.items():
            for i, _ in enumerate(self.observables):
                bit = int(key[::-1][i])  # qubit order reversed
                probs[i] += val * (1 if bit == 1 else -1)
        probs /= self.shots
        return probs

__all__ = ["HybridFCL"]
