import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import qiskit.circuit.random

class QuantumQuanvolutionFilter(nn.Module):
    def __init__(self, n_qubits: int = 8, backend=None, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        param_binds = []
        for sample in inputs:
            bind = {self.theta[i]: float(val) for i, val in enumerate(sample)}
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        probs = torch.zeros(self.n_qubits, device=inputs.device)
        for key, val in result.items():
            for i in range(self.n_qubits):
                if key[-(i + 1)] == '1':
                    probs[i] += val
        probs = probs / (self.shots * batch)
        return probs.unsqueeze(0).repeat(batch, 1)

def create_qcnn(num_qubits: int = 8) -> EstimatorQNN:
    feature_map = ZFeatureMap(num_qubits)
    params = ParameterVector("θ", length=num_qubits * 3)
    circuit = qiskit.QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        circuit.ry(params[i], i)
        circuit.rz(params[i + num_qubits], i)
        circuit.ry(params[i + 2 * num_qubits], i)
    circuit.measure_all()

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    estimator = Estimator()
    return EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=circuit.parameters,
        estimator=estimator
    )

class HybridBinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc_reduce = nn.Linear(540, 8)

        self.qfilter = QuantumQuanvolutionFilter()

        self.fc_concat = nn.Linear(16, 8)

        self.qcnn = create_qcnn()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc_reduce(x))

        qfeat = self.qfilter(x)

        concat = torch.cat([x, qfeat], dim=-1)
        concat = self.fc_concat(concat)

        probs = self.qcnn(concat.detach().cpu().numpy())
        probs = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        return probs.unsqueeze(-1)

__all__ = ["HybridBinaryClassifier"]
