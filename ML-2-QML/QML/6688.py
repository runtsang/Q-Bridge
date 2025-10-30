import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from typing import Sequence

class QCNNKernelHybridQML:
    """
    Quantum counterpart of QCNNKernelHybrid. Implements a variational
    circuit inspired by the QCNN architecture and a TorchQuantum
    ansatz for kernel evaluation.
    """
    def __init__(self, num_qubits: int = 8):
        algorithm_globals.random_seed = 12345
        self.estimator = Estimator()
        self.num_qubits = num_qubits

    def conv_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi/2, 0)
        return qc

    def pool_circuit(self, params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def conv_layer(self, num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self.conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self.conv_circuit(params[param_index:param_index+3]), [q1, q2])
            qc.barrier()
            param_index += 3
        return qc

    def pool_layer(self, sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits//2 * 3)
        for source, sink in zip(sources, sinks):
            qc.append(self.pool_circuit(params[param_index:param_index+3]), [source, sink])
            qc.barrier()
            param_index += 3
        return qc

    def build_ansatz(self):
        feature_map = ZFeatureMap(self.num_qubits)
        ansatz = QuantumCircuit(self.num_qubits, name="Ansatz")

        # First convolution and pooling
        ansatz.compose(self.conv_layer(self.num_qubits, "c1"), list(range(self.num_qubits)), inplace=True)
        ansatz.compose(self.pool_layer([0,1,2,3], [4,5,6,7], "p1"), list(range(self.num_qubits)), inplace=True)

        # Second convolution and pooling
        ansatz.compose(self.conv_layer(self.num_qubits//2, "c2"), list(range(self.num_qubits//2, self.num_qubits)), inplace=True)
        ansatz.compose(self.pool_layer([0,1], [2,3], "p2"), list(range(self.num_qubits//2, self.num_qubits)), inplace=True)

        # Third convolution and pooling
        ansatz.compose(self.conv_layer(self.num_qubits//4, "c3"), list(range(self.num_qubits//2, self.num_qubits)), inplace=True)
        ansatz.compose(self.pool_layer([0], [1], "p3"), list(range(self.num_qubits//2, self.num_qubits)), inplace=True)

        # Combine feature map and ansatz
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(feature_map, range(self.num_qubits), inplace=True)
        circuit.compose(ansatz, range(self.num_qubits), inplace=True)
        return circuit

    def get_qnn(self):
        circuit = self.build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits-1), 1)])
        qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=ZFeatureMap(self.num_qubits).parameters,
            weight_params=self.build_ansatz().parameters,
            estimator=self.estimator,
        )
        return qnn

    # TorchQuantum kernel interface
    class TorchQuantumKernal(tq.QuantumModule):
        def __init__(self, func_list):
            super().__init__()
            self.func_list = func_list

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor):
            q_device.reset_states(x.shape[0])
            for info in self.func_list:
                params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            for info in reversed(self.func_list):
                params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    class TorchQuantumKernel(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = QCNNKernelHybridQML.TorchQuantumKernal(
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

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        kernel = self.TorchQuantumKernel()
        return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QCNNKernelHybridQML"]
