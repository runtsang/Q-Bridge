import networkx as nx
import itertools
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, Sequence, List

class QuantumClassifierModel:
    """Quantum classifier with fidelity‑based graph regularisation and EstimatorQNN head."""
    def __init__(self, num_qubits: int, depth: int):
        self.circuit, self.encoding, self.weights, self.observables = \
            self.build_classifier_circuit(num_qubits, depth)

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
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        dataset = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = QuantumClassifierModel._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        target_unitary = QuantumClassifierModel._random_qubit_unitary(qnn_arch[-1])
        training_data = QuantumClassifierModel.random_training_data(target_unitary, samples)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = QuantumClassifierModel._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(QuantumClassifierModel._random_qubit_unitary(num_inputs + 1),
                                   QuantumClassifierModel._tensored_id(num_outputs - 1))
                    op = QuantumClassifierModel._swap_registers(op, num_inputs, num_inputs + output)
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
        return QuantumClassifierModel._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                       layer: int, input_state: qt.Qobj) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, QuantumClassifierModel._tensored_zero(num_outputs))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return QuantumClassifierModel._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs))

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        stored_states = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = QuantumClassifierModel._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = QuantumClassifierModel.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)
        circuit = QuantumCircuit(num_qubits)
        for param, qubit in zip(encoding, range(num_qubits)):
            circuit.rx(param, qubit)
        index = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                circuit.ry(weights[index], qubit)
                index += 1
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                       for i in range(num_qubits)]
        return circuit, list(encoding), list(weights), observables

    def run(self, input_values: List[float]) -> List[float]:
        """Execute the circuit with provided data and return expectation values."""
        param_dict = {str(p): val for p, val in zip(self.encoding, input_values)}
        bound_circuit = self.circuit.bind_parameters(param_dict)
        from qiskit.primitives import StatevectorSimulator
        simulator = StatevectorSimulator()
        result = simulator.run(bound_circuit).result()
        statevector = result.get_statevector(bound_circuit)
        expectations = []
        for obs in self.observables:
            expectations.append(float(statevector.expectation_value(obs)))
        return expectations
