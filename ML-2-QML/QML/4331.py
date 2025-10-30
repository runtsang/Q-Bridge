import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import qutip as qt
import scipy as sc
import networkx as nx
import itertools
import torchquantum as tq
from torchquantum.functional import func_name_dict

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
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list
        self.n_wires = len(func_list)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class HybridUnit:
    """Quantum counterpart of HybridUnit, providing quantum convolution, graph QNN, estimator, and kernel."""
    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 127,
        graph_threshold: float = 0.9,
        estimator_params: Sequence[Parameter] = None,
        kernel_func_list: Sequence[dict] = None,
    ) -> None:
        self.conv_kernel_size = conv_kernel_size
        self.conv_threshold = conv_threshold
        self.graph_threshold = graph_threshold

        # Quantum convolution circuit
        n_qubits = conv_kernel_size ** 2
        self._conv_circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self._conv_circuit.rx(self.theta[i], i)
        self._conv_circuit.barrier()
        self._conv_circuit += random_circuit(n_qubits, 2)
        self._conv_circuit.measure_all()
        self._conv_backend = qiskit.Aer.get_backend("qasm_simulator")
        self._conv_shots = 100

        # Estimator network
        if estimator_params is None:
            estimator_params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(estimator_params[0], 0)
        qc.rx(estimator_params[1], 0)
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[estimator_params[0]],
            weight_params=[estimator_params[1]],
            estimator=estimator,
        )

        # Kernel ansatz
        if kernel_func_list is None:
            kernel_func_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.kernel = Kernel(kernel_func_list)

    def conv(self, data: np.ndarray) -> float:
        """Run the quantum convolution circuit on 2D data."""
        data = np.reshape(data, (1, self.conv_kernel_size ** 2))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.conv_threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(
            self._conv_circuit,
            self._conv_backend,
            shots=self._conv_shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._conv_circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self._conv_shots * self.conv_kernel_size ** 2)

    def graph_fidelity(self, states: Sequence[qt.Qobj]) -> nx.Graph:
        """Build an adjacency graph based on quantum state fidelities."""
        def fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
            return abs((a.dag() * b)[0, 0]) ** 2

        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = fidelity(s_i, s_j)
            if fid >= self.graph_threshold:
                graph.add_edge(i, j, weight=1.0)
        return graph

    def estimate(self, input_val: float, weight_val: float) -> float:
        """Run the quantum estimator network."""
        return self.estimator_qnn(input_val, weight_val)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute quantum kernel matrix using TorchQuantum ansatz."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridUnit"]
