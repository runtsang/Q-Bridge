import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class ParameterizedQuantumCircuit:
    """Two‑qubit variational circuit with a single rotation angle."""
    def __init__(self, backend, shots=1024):
        self.backend = backend
        self.shots = shots
        self.theta = Parameter("theta")
        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.h([0, 1])
        self.circuit.rz(self.theta, 0)
        self.circuit.rz(-self.theta, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        exp = 0.0
        for state, count in result.items():
            bits = np.array(list(state[::-1]), dtype=int)
            parity = (-1) ** (bits[0] ^ bits[1])
            exp += parity * count
        exp /= self.shots
        return np.array([exp])

class QuantumHybridHead(nn.Module):
    """Forward a scalar through the variational circuit."""
    def __init__(self, backend, shots=1024, shift=np.pi/2):
        super().__init__()
        self.circuit = ParameterizedQuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thetas = x.detach().cpu().numpy() + self.shift
        expectations = self.circuit.run(thetas)
        return torch.tensor(expectations, device=x.device)

class QuantumRegressionHead(nn.Module):
    """Wraps the Qiskit EstimatorQNN for regression."""
    def __init__(self, backend):
        super().__init__()
        self.params = [Parameter("input1"), Parameter("weight1")]
        self.circuit = qiskit.QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.params[0], 0)
        self.circuit.rx(self.params[1], 0)
        self.circuit.measure_all()
        from qiskit.quantum_info import SparsePauliOp
        self.observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.params[0]],
            weight_params=[self.params[1]],
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_values = x.detach().cpu().numpy()
        outputs = self.estimator_qnn.predict(input_values)
        return torch.tensor(outputs, device=x.device)

class HybridQCNet(nn.Module):
    """Wrapper that replaces the classical heads with quantum ones."""
    def __init__(self, base_net: nn.Module, backend, shots=1024):
        super().__init__()
        self.base = base_net
        self.quantum_cls_head = QuantumHybridHead(backend, shots)
        self.quantum_reg_head = QuantumRegressionHead(backend)

    def forward(self, x: torch.Tensor):
        out = self.base(x)
        cls = self.quantum_cls_head(out["probability"])
        reg = self.quantum_reg_head(out["regression"])
        return {"probability": cls, "regression": reg}

__all__ = ["ParameterizedQuantumCircuit", "QuantumHybridHead", "QuantumRegressionHead", "HybridQCNet"]
