import numpy as np
import networkx as nx
import qutip as qt
import scipy as sc
import itertools
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import Sampler as QSampler, Estimator as QEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridConvNet:
    """
    Quantum hybrid convolutional network that builds a parameterised circuit
    mimicking a 2‑D convolution, provides a sampler and an estimator,
    and offers graph‑based fidelity utilities using qutip.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127.0,
                 shots: int = 100, backend=None):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.n_qubits = kernel_size ** 2
        self.circuit = self._build_circuit()
        self.sampler = QSampler()
        self.estimator = QEstimator()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(self.theta[i], i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the convolution‑like quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self.circuit, self.backend,
                      shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    # ------------------------------------------------------------------
    # Quantum Sampler and Estimator helpers
    # ------------------------------------------------------------------
    def sampler_qnn(self) -> QSamplerQNN:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        return QSamplerQNN(circuit=qc,
                           input_params=inputs,
                           weight_params=weights,
                           sampler=self.sampler)

    def estimator_qnn(self) -> QEstimatorQNN:
        params = [Parameter("input1"), Parameter("weight1")]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        observable = SparsePauliOp.from_list([("Y", 1)])
        return QEstimatorQNN(circuit=qc,
                             observables=observable,
                             input_params=[params[0]],
                             weight_params=[params[1]],
                             estimator=self.estimator)

    # ------------------------------------------------------------------
    # Graph‑based utilities (adapted from GraphQNN.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        identity = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        identity.dims = [dims.copy(), dims.copy()]
        return identity

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        projector = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        projector.dims = [dims.copy(), dims.copy()]
        return projector

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amplitudes /= sc.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
        dataset = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = HybridConvNet._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: list[int], samples: int):
        target_unitary = HybridConvNet._random_qubit_unitary(qnn_arch[-1])
        training_data = HybridConvNet.random_training_data(target_unitary, samples)

        unitaries: list[list[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: list[qt.Qobj] = []
            for output in range(num_outputs):
                op = HybridConvNet._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(HybridConvNet._random_qubit_unitary(num_inputs + 1),
                                   HybridConvNet._tensored_id(num_outputs - 1))
                    op = HybridConvNet._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: list[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: list[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return HybridConvNet._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(qnn_arch: list[int], unitaries: list[list[qt.Qobj]],
                       layer: int, input_state: qt.Qobj) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, HybridConvNet._tensored_zero(num_outputs))

        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary

        return HybridConvNet._partial_trace_remove(layer_unitary * state * layer_unitary.dag(),
                                                   range(num_inputs))

    @staticmethod
    def feedforward(qnn_arch: list[int], unitaries: list[list[qt.Qobj]],
                    samples: list[tuple[qt.Qobj, qt.Qobj]]):
        stored_states = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = HybridConvNet._layer_channel(qnn_arch,
                                                             unitaries,
                                                             layer,
                                                             current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: list[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridConvNet.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["HybridConvNet"]
