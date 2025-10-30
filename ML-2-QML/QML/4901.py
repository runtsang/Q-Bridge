import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

def _tensored_id(num_qubits: int):
    return qiskit.quantum_info.Operator(qiskit.quantum_info.QubitOperator('z'*num_qubits))

def _random_qubit_unitary(num_qubits: int):
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return q

def _random_qubit_state(num_qubits: int):
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

def random_training_data(unitary: Statevector, samples: int):
    data = []
    n = unitary.num_qubits
    for _ in range(samples):
        state = _random_qubit_state(n)
        data.append((state, unitary @ state))
    return data

def random_network(arch: List[int], samples: int):
    """
    Build a layer‑wise stack of random unitaries and produce a synthetic
    training set using the final unitary as target.
    """
    num_layers = len(arch) - 1
    unitaries: List[List[Statevector]] = [[]]
    for layer in range(1, len(arch)):
        n_in, n_out = arch[layer-1], arch[layer]
        layer_ops: List[Statevector] = []
        for _ in range(n_out):
            mat = _random_qubit_unitary(n_in + 1)
            layer_ops.append(Statevector(mat))
        unitaries.append(layer_ops)
    target = _random_qubit_unitary(arch[-1])
    data = random_training_data(Statevector(target), samples)
    return arch, unitaries, data, Statevector(target)

def _layer_channel(arch: Sequence[int], unitaries: Sequence[Sequence[Statevector]], layer: int, state: Statevector):
    """
    Apply a single layer of the graph‑quantum network to a state.
    """
    n_in = arch[layer-1]
    n_out = arch[layer]
    # pad state with |0⟩ on new qubits
    padded = Statevector.tensor(state, Statevector([1,0]))
    # apply each unitary in the layer
    out_state = padded
    for gate in unitaries[layer]:
        out_state = gate @ out_state
    # trace out the input registers
    return out_state.partial_trace(range(n_in))

def feedforward(arch: Sequence[int], unitaries: Sequence[Sequence[Statevector]], samples: Iterable[Tuple[Statevector, Statevector]]):
    """
    Return layer‑wise states for each sample.
    """
    all_states = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layerwise.append(current)
        all_states.append(layerwise)
    return all_states

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """
    Absolute‑squared overlap of two pure states.
    """
    return float(abs(np.vdot(a.data, b.data)) ** 2)

def fidelity_adjacency(states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    """
    Build a weighted graph from state fidelities.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

class GraphQNN:
    """
    Quantum graph neural network mirroring the classical interface.
    Uses qiskit for circuit creation, statevector simulation for forward
    propagation, and a fidelity‑based adjacency graph.
    """

    def __init__(self, arch: Sequence[int], backend=None, shots=1024):
        self.arch = list(arch)
        self.backend = backend or qiskit.Aer.get_backend("statevector_simulator")
        self.shots = shots
        self.circuits: List[QuantumCircuit] = []
        # build a circuit per output node in each layer
        for layer in range(1, len(arch)):
            n_in, n_out = arch[layer-1], arch[layer]
            for out_idx in range(n_out):
                qc = QuantumCircuit(n_in + 1, name=f"layer{layer}_out{out_idx}")
                # random unitary via rotation angles
                for qubit in range(n_in + 1):
                    qc.h(qubit)
                    theta = qiskit.circuit.Parameter(f"theta_{layer}_{out_idx}_{qubit}")
                    qc.rx(theta, qubit)
                qc.measure_all()
                self.circuits.append(qc)

    def _execute_batch(self, inputs: List[np.ndarray]) -> List[Statevector]:
        """
        Execute the circuit stack on a batch of input vectors.
        """
        results: List[Statevector] = []
        for qc in self.circuits:
            # bind parameters to a fixed random instance per batch
            bound = qc.copy()
            for idx, param in enumerate(bound.parameters):
                bound.assign_parameters({param: np.random.rand()}, inplace=True)
            job = execute(bound, self.backend, shots=self.shots)
            vec = Statevector.from_instruction(bound)
            results.append(vec)
        return results

    def forward(self, inputs: List[np.ndarray]) -> List[Statevector]:
        """
        Propagate each input through the quantum network and return the
        final layer states.
        """
        return self._execute_batch(inputs)

    def fidelity_adjacency(
        self,
        samples: Iterable[Tuple[Statevector, Statevector]],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Generate a weighted graph from the fidelities of the final output states.
        """
        final_states = [y for _, y in samples]
        return fidelity_adjacency(final_states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def train(self, data: Iterable[Tuple[Statevector, Statevector]], epochs: int = 10):
        """
        Placeholder for a training routine that would optimise the circuit
        parameters.  In practice one would use a Qiskit‑Machine‑Learning
        EstimatorQNN or a custom gradient‑based optimiser.
        """
        pass
