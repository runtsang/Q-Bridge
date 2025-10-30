import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumConvFilter:
    """Quantum implementation of a 2x2 convolution filter."""
    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: int = 127):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data):
        """Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class QuantumRegressor(nn.Module):
    """Hybrid regressor using Qiskit's EstimatorQNN."""
    def __init__(self, backend=None):
        super().__init__()
        self.backend = backend or Aer.get_backend("statevector_simulator")
        # Simple 1‑qubit circuit
        param_input = Parameter("input")
        param_weight = Parameter("weight")
        qc = qiskit.QuantumCircuit(1)
        qc.h(0)
        qc.ry(param_input, 0)
        qc.rx(param_weight, 0)
        from qiskit.quantum_info import SparsePauliOp
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[param_input],
            weight_params=[param_weight],
            estimator=estimator
        )

    def forward(self, x):
        # x: Tensor of shape (batch, 1)
        expectations = []
        for val in x.squeeze().tolist():
            expectation = self.estimator_qnn.predict([(val, 0.0)])[0]
            expectations.append(expectation)
        return torch.tensor(expectations, dtype=torch.float32).unsqueeze(-1)

class QuantumHybridHead(nn.Module):
    """Hybrid expectation head for binary classification."""
    def __init__(self,
                 n_qubits: int = 1,
                 backend=None,
                 shots: int = 100,
                 shift: float = np.pi / 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.shift = shift
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

    def run(self, x):
        # x: Tensor of shape (batch, 1)
        expectations = []
        for val in x.squeeze().tolist():
            bind = {self.theta: val}
            compiled = transpile(self._circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts(self._circuit)
            # expectation of Z
            exp = 0
            for key, cnt in counts.items():
                z = 1 if key[-1] == '1' else -1
                exp += z * cnt
            exp /= self.shots
            expectations.append(exp)
        return torch.tensor(expectations, dtype=torch.float32).unsqueeze(-1)

class ConvHybridNet(nn.Module):
    """Quantum‑enhanced convolutional network for regression or binary classification.

    The network replaces the classical Conv filter with a parameterised quantum circuit
    and the head with either a quantum estimator (regression) or a hybrid expectation
    block (classification).  The design is inspired by Conv, EstimatorQNN,
    ClassicalQuantumBinaryClassification, and FraudDetection.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: int = 127,
                 regression: bool = True,
                 classification: bool = False,
                 backend=None,
                 shots: int = 100):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.regression = regression
        self.classification = classification
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.conv = QuantumConvFilter(kernel_size=kernel_size,
                                      backend=self.backend,
                                      shots=shots,
                                      threshold=threshold)
        if regression:
            self.head = QuantumRegressor(backend=self.backend)
        elif classification:
            self.head = QuantumHybridHead(n_qubits=1,
                                          backend=self.backend,
                                          shots=shots)
        else:
            self.head = nn.Identity()

    def forward(self, data):
        """
        Args:
            data: Tensor of shape (batch, kernel_size, kernel_size)
        Returns:
            Tensor of shape (batch, 1) for regression,
            or (batch, 2) for binary classification.
        """
        conv_out = []
        for img in data:
            val = self.conv.run(img.numpy())
            conv_out.append(val)
        conv_out = torch.tensor(conv_out, dtype=torch.float32).unsqueeze(-1)
        out = self.head(conv_out)
        if self.classification:
            prob = torch.sigmoid(out)
            return torch.cat([prob, 1 - prob], dim=-1)
        return out

__all__ = ["ConvHybridNet"]
