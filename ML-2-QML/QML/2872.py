import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp

class QuantumQCNN:
    """Parameterized QCNN circuit built from convolution and pooling subcircuits."""
    def __init__(self, backend: AerSimulator, shots: int = 100):
        self.backend = backend
        self.shots = shots
        self.num_qubits = 8
        self.circuit = self._build_circuit()
        self.num_params = 42  # 12+12+6+6+3+3
        self.param_vector = ParameterVector("θ", length=self.num_params)

    # ----- Sub‑circuit builders -------------------------------------------------
    def _conv_circuit(self, params, q1, q2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        return sub

    def _pool_circuit(self, params, q1, q2):
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        param_index = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            sub = self._conv_circuit(params[param_index:param_index+3], q1, q2)
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            sub = self._conv_circuit(params[param_index:param_index+3], q1, q2)
            qc.append(sub, [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        param_index = 0
        for source, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[param_index:param_index+3], source, sink)
            qc.append(sub, [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    def _build_circuit(self):
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8)
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4,8)), inplace=True)
        ansatz.compose(self._pool_layer([0,1], [2,3], "p2"), list(range(4,8)), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6,8)), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6,8)), inplace=True)
        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit

    # ----- Execution ------------------------------------------------------------
    def run(self, param_values):
        """Execute the QCNN for a batch of parameter vectors."""
        if isinstance(param_values, list):
            param_values = np.array(param_values)
        batch_size = param_values.shape[0]
        expectations = []
        for i in range(batch_size):
            param_bind = {self.param_vector[j]: param_values[i, j] for j in range(self.num_params)}
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[param_bind])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            # Expectation value of Z on the first qubit
            exp = 0.0
            for state, cnt in counts.items():
                bit = int(state[0])  # first qubit
                prob = cnt / self.shots
                exp += (1 if bit == 0 else -1) * prob
            expectations.append(exp)
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """Differentiable bridge to the quantum QCNN."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_qcnn: QuantumQCNN, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_qcnn = quantum_qcnn
        expectations = quantum_qcnn.run(inputs.tolist())
        result = torch.tensor(expectations, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grads = []
        for i in range(inputs.shape[0]):
            grad_row = []
            for j in range(inputs.shape[1]):
                param = inputs[i, j].item()
                plus = ctx.quantum_qcnn.run([[param + shift]])[0]
                minus = ctx.quantum_qcnn.run([[param - shift]])[0]
                grad_row.append(plus - minus)
            grads.append(grad_row)
        grads = torch.tensor(grads, dtype=torch.float32)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through the quantum QCNN."""
    def __init__(self, quantum_qcnn: QuantumQCNN, shift: float):
        super().__init__()
        self.quantum_qcnn = quantum_qcnn
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_qcnn, self.shift)

class HybridQCNNNet(nn.Module):
    """Complete hybrid classifier combining classical CNN, dense layers, and a QCNN quantum head."""
    def __init__(self):
        super().__init__()
        # Classical feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Dense backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Map to QCNN input dimension
        self.fc_qcnn = nn.Linear(1, 42)
        backend = AerSimulator()
        self.quantum_qcnn = QuantumQCNN(backend=backend, shots=100)
        self.hybrid = Hybrid(self.quantum_qcnn, shift=np.pi / 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
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
        x = self.fc_qcnn(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["QuantumQCNN", "HybridFunction", "Hybrid", "HybridQCNNNet"]
