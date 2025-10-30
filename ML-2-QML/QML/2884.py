import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import torch
from torch import nn
from typing import Iterable

class HybridFCLEstimator:
    """
    Quantum‑classical hybrid model that combines a classical pre‑processing
    network with a qiskit EstimatorQNN quantum layer. The classical part
    maps input features to a weight parameter that controls the quantum
    rotation, while the EstimatorQNN evaluates the expectation of a
    single‑qubit observable.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        # Classical pre‑processing network
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )
        # Quantum circuit
        theta = Parameter("theta")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        # Observable Y
        from qiskit.quantum_info import SparsePauliOp
        observable = SparsePauliOp.from_list([("Y", 1)])
        # Estimator
        estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[],
            weight_params=[theta],
            estimator=estimator,
        )
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre_net(x)

    def run(self, inputs: np.ndarray, thetas: Iterable[float]) -> np.ndarray:
        # Classical pre‑processing
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        weight = self.forward(inp).detach().numpy().flatten()
        # Quantum evaluation
        job = execute(
            self.qnn.circuit,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=[{self.qnn.weight_params[0]: w} for w in weight],
        )
        result = job.result().get_counts(self.qnn.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / self.shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridFCLEstimator"]
