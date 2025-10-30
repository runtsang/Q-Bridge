import qiskit.circuit as circuit
import qiskit.primitives as primitives
import qutip as qt
import networkx as nx
import itertools
import numpy as np
from typing import Iterable, Sequence, Tuple, List

__all__ = ["SamplerQNNGen256"]

class SamplerQNNGen256:
    """
    Quantum sampler mirroring the classical SamplerQNNGen256.

    Parameters
    ----------
    qnn_arch : Sequence[int], optional
        Layer sizes for the quantum circuit. Default is (2, 4, 2).
    circuit : qiskit.circuit.QuantumCircuit, optional
        Pre‑built parameterized circuit.
    input_params : qiskit.circuit.ParameterVector, optional
        Input parameters for the circuit.
    weight_params : qiskit.circuit.ParameterVector, optional
        Weight parameters for the circuit.
    sampler : qiskit.primitives.Sampler, optional
        Quantum sampler primitive.
    """
    def __init__(self, qnn_arch: Sequence[int] = (2, 4, 2),
                 circuit: circuit.QuantumCircuit | None = None,
                 input_params: circuit.ParameterVector | None = None,
                 weight_params: circuit.ParameterVector | None = None,
                 sampler: primitives.Sampler | None = None):
        self.qnn_arch = qnn_arch
        self.circuit = circuit
        self.input_params = input_params
        self.weight_params = weight_params
        self.sampler = sampler

        if self.circuit is None:
            # Build a simple Ry–CX–Ry circuit with 2 qubits
            self.input_params = circuit.ParameterVector("input", 2)
            self.weight_params = circuit.ParameterVector("weight", 4)
            self.circuit = circuit.QuantumCircuit(2)
            self.circuit.ry(self.input_params[0], 0)
            self.circuit.ry(self.input_params[1], 1)
            self.circuit.cx(0, 1)
            self.circuit.ry(self.weight_params[0], 0)
            self.circuit.ry(self.weight_params[1], 1)
            self.circuit.cx(0, 1)
            self.circuit.ry(self.weight_params[2], 0)
            self.circuit.ry(self.weight_params[3], 1)

        if self.sampler is None:
            self.sampler = primitives.Sampler()

    def sample(self, inputs: np.ndarray | None = None, **kwargs):
        """
        Run the quantum sampler and return a probability distribution.

        Parameters
        ----------
        inputs : np.ndarray, optional
            Input parameter values. If None, random values are used.
        """
        if inputs is None:
            inputs = np.random.rand(len(self.input_params))
        param_dict = {p: v for p, v in zip(self.input_params, inputs)}
        if self.weight_params is not None:
            weight_vals = np.random.rand(len(self.weight_params))
            param_dict.update({p: v for p, v in zip(self.weight_params, weight_vals)})
        result = self.sampler.run(self.circuit, param_dict, **kwargs)
        return result

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int = 10):
        """
        Generate a random quantum circuit and training data.

        Returns
        -------
        qnn_arch : Sequence[int]
            Architecture of the network.
        unitaries : List[List[qt.Qobj]]
            List of layers, each containing a list of unitary gates.
        training_data : List[Tuple[qt.Qobj, qt.Qobj]]
            Pairs of input and target states.
        target_unitary : qt.Qobj
            The unitary representing the target transformation.
        """
        target_unitary = SamplerQNNGen256._random_qubit_unitary(qnn_arch[-1])
        training_data = SamplerQNNGen256._random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = SamplerQNNGen256._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(SamplerQNNGen256._random_qubit_unitary(num_inputs + 1),
                                   qt.qeye(2 ** (num_outputs - 1)))
                    op = SamplerQNNGen256._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]):
        """Propagate a set of input states through the quantum network."""
        stored_states = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = SamplerQNNGen256._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Overlap squared between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = SamplerQNNGen256.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        unitary, _ = np.linalg.qr(matrix)
        qobj = qt.Qobj(unitary)
        qobj.dims = [[2] * num_qubits, [2] * num_qubits]
        return qobj

    @staticmethod
    def _random_training_data(unitary: qt.Qobj, samples: int):
        data = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = SamplerQNNGen256._random_qubit_state(num_qubits)
            data.append((state, unitary * state))
        return data

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = np.random.randn(dim) + 1j * np.random.randn(dim)
        amplitudes /= np.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                       layer: int, input_state: qt.Qobj) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, qt.qeye(2 ** num_outputs))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return qt.ptrace(layer_unitary * state * layer_unitary.dag(), list(range(num_inputs)))
