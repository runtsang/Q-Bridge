import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN


class QuantumCircuitWrapper:
    """Thin wrapper that executes a parametrised circuit on a simulator."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.theta = theta

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[{self.theta: t} for t in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        exp = ctx.circuit.run(inputs.tolist())
        return torch.tensor(exp, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors  # not used directly
        shift = ctx.shift
        grads = []
        for val in inputs.tolist():
            exp_r = ctx.circuit.run([val + shift])
            exp_l = ctx.circuit.run([val - shift])
            grads.append(exp_r - exp_l)
        return torch.tensor(grads, dtype=torch.float32) * grad_output, None, None


class HybridHead(nn.Module):
    """Quantum‑based head that replaces the classical sigmoid."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.shift)


class EstimatorQNN(nn.Module):
    """
    Quantum‑enhanced estimator that mirrors the classical EstimatorQNN.
    The circuit is a compact QCNN‑style ansatz composed of a Z‑feature map
    followed by a single convolutional layer of two‑qubit blocks.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 shots: int = 100,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.estimator = StatevectorEstimator()

        # Feature map
        self.feature_map = ZFeatureMap(n_qubits)

        # Build a single convolutional block
        def conv_block(num_qubits: int, param_prefix: str) -> qiskit.QuantumCircuit:
            qc = qiskit.QuantumCircuit(num_qubits)
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for i in range(0, num_qubits, 2):
                sub = qiskit.QuantumCircuit(2)
                sub.rz(-np.pi / 2, i + 1)
                sub.cx(i + 1, i)
                sub.rz(params[i], i)
                sub.ry(params[i + 1], i + 1)
                sub.cx(i, i + 1)
                sub.ry(params[i + 2], i + 1)
                sub.cx(i + 1, i)
                sub.rz(np.pi / 2, i)
                qc.append(sub, [i, i + 1])
            return qc

        ansatz = qiskit.QuantumCircuit(n_qubits)
        ansatz.compose(conv_block(n_qubits, "c1"), inplace=True)

        # Assemble full circuit
        full_circuit = qiskit.QuantumCircuit(n_qubits)
        full_circuit.compose(self.feature_map, range(n_qubits), inplace=True)
        full_circuit.compose(ansatz, range(n_qubits), inplace=True)

        # EstimatorQNN from qiskit_machine_learning
        self.qnn = QiskitEstimatorQNN(
            circuit=full_circuit.decompose(),
            observables=SparsePauliOp.from_list([("Z" * n_qubits, 1)]),
            input_params=self.feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, features). Features are interpreted as expectation values
            for the feature‑map parameters.
        """
        return self.qnn(x).T
