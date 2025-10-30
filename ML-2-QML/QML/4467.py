import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import qiskit
from qiskit import Aer, execute
from qiskit.circuit.random import random_circuit

class QuantumSelfAttentionCircuit:
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(n_qubits, 2)
        self.circuit.measure_all()
    def run(self, params: np.ndarray) -> np.ndarray:
        exp_list = []
        for val in params:
            param_bind = {self.theta[0]: val}
            job = execute(self.circuit, self.backend, shots=self.shots,
                          parameter_binds=[param_bind])
            result = job.result().get_counts()
            exp = 0.0
            for key, val in result.items():
                bit = int(key[0])
                exp += ((-1) ** bit) * val
            exp /= self.shots
            exp_list.append(exp)
        return np.array(exp_list)

class QuantumConvCircuit:
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: int = 127):
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()
    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

class HybridHead(nn.Module):
    def __init__(self, in_features: int, n_qubits: int = 1, shots: int = 200, shift: float = np.pi / 2):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.shots = shots
        self.shift = shift
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(self.theta)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
    def run(self, input_val: float) -> np.ndarray:
        param_bind = {self.theta: input_val}
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[param_bind])
        result = job.result().get_counts()
        exp = 0.0
        for key, val in result.items():
            bit = int(key[0])
            exp += ((-1) ** bit) * val
        exp /= self.shots
        return np.array([exp])
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_np = input_tensor.detach().cpu().float().numpy().flatten()
        outputs = [self.run(val) for val in input_np]
        return torch.tensor(outputs, dtype=torch.float32)

class QuantumTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        job = execute(self.circuit, self.backend, shots=self.shots)
        result = job.result().get_counts()
        exp = sum(int(bit) * val for bit, val in result.items()) / (self.shots * 1)
        return torch.tensor([exp], dtype=torch.float32)

class UnifiedHybridClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, use_transformer: bool = True):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1)
        )
        self.image_self_attn = QuantumSelfAttentionCircuit(n_qubits=1)
        self.image_conv_filter = QuantumConvCircuit(kernel_size=2)
        self.use_transformer = use_transformer
        if use_transformer:
            self.text_branch = QuantumTransformer()
        else:
            self.text_branch = nn.Sequential(
                nn.Embedding(30522, 64),
                nn.Flatten(),
                nn.Linear(64 * 10, 1)
            )
        self.classifier = nn.Linear(2, num_classes if num_classes > 2 else 1)
    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_branch(image)
        img_feat_np = img_feat.detach().cpu().numpy().flatten()
        attn_out = self.image_self_attn.run(img_feat_np)
        conv_out = self.image_conv_filter.run(attn_out)
        img_feat = torch.tensor([conv_out], dtype=torch.float32)
        txt_feat = self.text_branch(text)
        combined = torch.cat([img_feat, txt_feat], dim=-1)
        out = self.classifier(combined)
        return torch.cat([out, 1 - out], dim=-1)
