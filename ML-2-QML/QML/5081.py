import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit import ParameterVector, Parameter
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QuantumConvLayer:
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100,
                 threshold: float = 0.0, clip: bool = True) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.clip = clip
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        thetas = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, theta in enumerate(thetas):
            qc.rx(theta, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in data:
            bind = {}
            for i, val in enumerate(row):
                theta_val = np.pi if val > self.threshold else 0.0
                if self.clip:
                    theta_val = _clip(theta_val, np.pi)
                bind[self.circuit.parameters[i]] = theta_val
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        ones_sum = sum(
            sum((1 - 2 * int(bit)) for bit in bitstring) * count
            for bitstring, count in result.items()
        )
        return ones_sum / (self.shots * self.n_qubits)

class QuantumFullyConnectedLayer:
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100, clip: bool = True) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.clip = clip
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        theta = Parameter("theta")
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray) -> float:
        theta_val = thetas[0] if thetas.size > 0 else 0.0
        if self.clip:
            theta_val = _clip(theta_val, np.pi)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[{self.circuit.parameters[0]: theta_val}])
        result = job.result().get_counts(self.circuit)
        ones_sum = sum(
            sum((1 - 2 * int(bit)) for bit in bitstring) * count
            for bitstring, count in result.items()
        )
        return ones_sum / (self.shots * self.n_qubits)

def conv_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index:param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for src, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index:param_index + 3]), [src, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

class QuantumQCNNLayer:
    def __init__(self, num_qubits: int = 8, backend=None, shots: int = 100, clip: bool = True) -> None:
        self.num_qubits = num_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.clip = clip
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(conv_layer(self.num_qubits, "c1"), range(self.num_qubits), inplace=True)
        qc.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"),
                   range(self.num_qubits), inplace=True)
        qc.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
        qc.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        qc.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
        qc.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        qc.measure_all()
        return qc

    def run(self, thetas: np.ndarray) -> float:
        bind_dict = {p: thetas[0] if thetas.size > 0 else 0.0 for p in self.circuit.parameters}
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=[bind_dict])
        result = job.result().get_counts(self.circuit)
        ones_sum = sum(
            sum((1 - 2 * int(bit)) for bit in bitstring) * count
            for bitstring, count in result.items()
        )
        return ones_sum / (self.shots * self.num_qubits)

class HybridGen445:
    def __init__(self, kernel_size: int = 2, num_qubits: int = 8,
                 backend=None, shots: int = 100, threshold: float = 0.0) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.conv = QuantumConvLayer(kernel_size=kernel_size,
                                     backend=self.backend,
                                     shots=shots,
                                     threshold=threshold)
        self.fcl = QuantumFullyConnectedLayer(n_qubits=num_qubits,
                                              backend=self.backend,
                                              shots=shots)
        self.qcnn = QuantumQCNNLayer(num_qubits=num_qubits,
                                     backend=self.backend,
                                     shots=shots)
        self.fraud_params = self._random_fraud_params()

    def _random_fraud_params(self) -> FraudLayerParameters:
        return FraudLayerParameters(
            bs_theta=np.random.uniform(-np.pi, np.pi),
            bs_phi=np.random.uniform(-np.pi, np.pi),
            phases=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            squeeze_r=(np.random.uniform(-3, 3), np.random.uniform(-3, 3)),
            squeeze_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            displacement_r=(np.random.uniform(-3, 3), np.random.uniform(-3, 3)),
            displacement_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            kerr=(np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
        )

    def run(self, data: np.ndarray) -> float:
        conv_out = self.conv.run(data)
        fcl_out = self.fcl.run(np.array([conv_out]))
        qcnn_out = self.qcnn.run(np.array([fcl_out]))
        return conv_out + fcl_out + qcnn_out

__all__ = ["HybridGen445", "FraudLayerParameters"]
