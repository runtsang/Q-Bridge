from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable, Sequence
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp, BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator

class HybridQCNet(nn.Module):
    """Hybrid CNN followed by a quantum expectation head."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 n_qubits: int = 2, shots: int = 200) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.backend = Aer.get_backend("aer_simulator")
        self.n_qubits = n_qubits
        self.shots = shots
        self._build_quantum_head()

    def _build_quantum_head(self) -> None:
        self.qc = QuantumCircuit(self.n_qubits)
        self.theta = Parameter("theta")
        self.qc.h(range(self.n_qubits))
        self.qc.ry(self.theta, range(self.n_qubits))
        self.qc.measure_all()
        self.estimator = Estimator()
        self.observable = SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])

    def _expectation(self, params: np.ndarray) -> float:
        bound = self.qc.assign_parameters({self.theta: params[0]})
        state = Statevector.from_instruction(bound)
        return float(state.expectation_value(self.observable))

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
        logits = self.fc3(x).squeeze(-1)
        logits_np = logits.detach().cpu().numpy()
        exp_vals = np.array([self._expectation(np.array([v])) for v in logits_np])
        probs = 1 / (1 + np.exp(-exp_vals))
        probs_t = torch.from_numpy(probs).to(logits.device)
        return torch.cat((probs_t, 1 - probs_t), dim=-1)

class EstimatorQNN:
    """Quantum regression network using Qiskit Machine Learning EstimatorQNN."""
    def __init__(self) -> None:
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = Estimator()
        self.qnn = QEstimatorQNN(circuit=qc,
                                 observables=observable,
                                 input_params=[params[0]],
                                 weight_params=[params[1]],
                                 estimator=estimator)

    def __call__(self, *args, **kwargs):
        return self.qnn(*args, **kwargs)

class FCL:
    """Parameterised quantum circuit emulating a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self._build_circuit()

    def _build_circuit(self) -> None:
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = Parameter("theta")
        self.circuit.h(range(self.n_qubits))
        self.circuit.ry(self.theta, range(self.n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[{self.theta: t} for t in thetas])
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        states = np.array(list(result.keys())).astype(float)
        return np.sum(states * probs)

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, param_vals: Sequence[float]) -> QuantumCircuit:
        mapping = dict(zip(self.params, param_vals))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> list[list[complex]]:
        results = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            state = Statevector.from_instruction(bound)
            results.append([state.expectation_value(obs) for obs in observables])
        return results

__all__ = ["HybridQCNet", "EstimatorQNN", "FCL", "FastBaseEstimator"]
