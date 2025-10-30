"""Quantum classifier model using TorchQuantum data‑re‑uploading and a parameterised kernel."""
import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List, Sequence
import torchquantum as tq
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list: List[dict]) -> None:
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

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class FCLQuantum:
    """Simple parameterised quantum circuit for a fully‑connected layer."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024) -> None:
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class QuantumClassifierModelGen:
    """Quantum classifier with data‑re‑uploading, a quantum kernel, and optional FCL."""
    def __init__(self, num_qubits: int, depth: int = 3, use_fcl: bool = False) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_fcl = use_fcl

        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit()

        if self.use_fcl:
            self.fcl = FCLQuantum(num_qubits)
        else:
            self.fcl = None

    def build_classifier_circuit(self) -> Tuple[tq.QuantumCircuit, Iterable, Iterable, List[tq.SparsePauliOp]]:
        encoding = tq.ParameterVector("x", self.num_qubits)
        weights = tq.ParameterVector("theta", self.num_qubits * self.depth)

        circuit = tq.QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [tq.SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1)) for i in range(self.num_qubits)]
        return circuit, list(encoding), list(weights), observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.circuit.reset_states(x.shape[0])
        self.circuit.apply(x)
        # Return expectation of Z on the first qubit as a simple output
        return self.circuit.expectation([tq.SparsePauliOp("Z")])[0]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-3) -> None:
        optimizer = torch.optim.Adam(self.circuit.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.circuit.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = self.forward(X_t)
            return preds.argmax(dim=-1).cpu().numpy()
