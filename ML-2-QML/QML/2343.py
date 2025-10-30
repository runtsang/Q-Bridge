import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Sequence

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

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

    def matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

class QCNNHybrid:
    """Hybrid quantum‑classical QCNN with a TorchQuantum kernel.

    The class builds a variational ansatz circuit (convolution + pooling)
    and a feature map. It exposes an EstimatorQNN for inference and
    a quantum kernel for similarity computations.
    """
    def __init__(self, seed: int = 12345):
        algorithm_globals.random_seed = seed
        self.estimator = Estimator()
        self.circuit = self._build_ansatz()
        self.feature_map = ZFeatureMap(8)
        # Combine feature map and ansatz
        self.full_circuit = QuantumCircuit(8)
        self.full_circuit.compose(self.feature_map, range(8), inplace=True)
        self.full_circuit.compose(self.circuit, range(8), inplace=True)
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.qnn = EstimatorQNN(
            circuit=self.full_circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )
        self.kernel = Kernel()

    def _conv_circuit(self, params):
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

    def _conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(self._conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def _pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(self._pool_circuit(params[param_index:param_index+3]), [source, sink])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    def _build_ansatz(self):
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)
        return ansatz

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the EstimatorQNN to obtain predictions."""
        return self.qnn.predict(inputs.numpy())

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the TorchQuantum kernel."""
        return self.kernel.matrix(a, b)

def QCNNHybridFactory(seed: int = 12345) -> QCNNHybrid:
    """Factory returning a configured :class:`QCNNHybrid` instance."""
    return QCNNHybrid(seed)

__all__ = ["KernalAnsatz", "Kernel", "QCNNHybrid", "QCNNHybridFactory"]
