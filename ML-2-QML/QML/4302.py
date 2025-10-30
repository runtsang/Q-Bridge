import networkx as nx
import qutip as qt
import scipy as sc
import itertools
from typing import List, Tuple, Iterable, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class HybridSamplerQNN:
    """
    Quantum counterpart to the classical HybridSamplerQNN.  It contains:

    1. A parameterised sampler circuit that can be executed with a
       StatevectorSampler.
    2. A graph‑based quantum neural network API that mirrors the
       classical GraphQNN utilities.
    3. A simple variational classifier circuit with Pauli‑Z observables.

    All components are fully self‑contained and can be instantiated
    independently.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        graph_arch: Sequence[int] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.graph_arch = graph_arch or (2, 2, 2)
        (
            self.sampler_circuit,
            self.sampler,
            self.input_params,
            self.weight_params,
        ) = self._build_sampler()
        (
            self.classifier_circuit,
            self.enc_params,
            self.cls_weights,
            self.observables,
        ) = self._build_classifier()

    # ------------------------------------------------------------------
    # 1. sampler circuit
    # ------------------------------------------------------------------
    def _build_sampler(
        self,
    ) -> Tuple[QuantumCircuit, SamplerQNN, ParameterVector, ParameterVector]:
        inputs = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(inputs, range(self.num_qubits)):
            qc.ry(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        sampler = StatevectorSampler()
        sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )
        return qc, sampler_qnn, inputs, weights

    # ------------------------------------------------------------------
    # 2. classifier circuit
    # ------------------------------------------------------------------
    def _build_classifier(
        self,
    ) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    # ------------------------------------------------------------------
    # 3. sampling
    # ------------------------------------------------------------------
    def sample(self, input_values: List[float]) -> qt.Qobj:
        bound = self.sampler_circuit.bind_parameters(
            dict(zip(self.input_params, input_values))
        )
        statevec = Statevector(bound)
        return qt.Qobj(statevec.data)

    # ------------------------------------------------------------------
    # 4. classification
    # ------------------------------------------------------------------
    def classify(self, input_values: List[float]) -> List[float]:
        bound = self.classifier_circuit.bind_parameters(
            dict(zip(self.enc_params, input_values))
        )
        statevec = Statevector(bound)
        return [float(statevec.expectation_value(obs).real) for obs in self.observables]

    # ------------------------------------------------------------------
    # 5. graph‑based quantum utilities
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
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(
            size=(dim, dim)
        )
        unitary = sc.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(
            size=(dim, 1)
        )
        amplitudes /= sc.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_training_data(
        unitary: qt.Qobj, samples: int
    ) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = HybridSamplerQNN._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(
        qnn_arch: List[int], samples: int = 10
    ) -> Tuple[
        List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj
    ]:
        target_unitary = HybridSamplerQNN._random_qubit_unitary(qnn_arch[-1])
        training_data = HybridSamplerQNN.random_training_data(target_unitary, samples)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = HybridSamplerQNN._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(
                        HybridSamplerQNN._random_qubit_unitary(num_inputs + 1),
                        HybridSamplerQNN._tensored_id(num_outputs - 1),
                    )
                    op = HybridSamplerQNN._swap_registers(
                        op, num_inputs, num_inputs + output
                    )
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return HybridSamplerQNN._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, HybridSamplerQNN._tensored_zero(num_outputs))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return HybridSamplerQNN._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs)
        )

    def feedforward(
        self,
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise: List[qt.Qobj] = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = self._layer_channel(
                    qnn_arch, unitaries, layer, current_state
                )
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def compute_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(
            enumerate(states), 2
        ):
            fid = HybridSamplerQNN.compute_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["HybridSamplerQNN"]
