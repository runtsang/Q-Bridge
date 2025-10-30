import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import networkx as nx
import numpy as np
from typing import Iterable, Sequence, List, Tuple

def _tensored_id(num_qubits: int):
    return qiskit.quantum_info.Operator(np.eye(2 ** num_qubits))

def _tensored_zero(num_qubits: int):
    zero = np.zeros((2 ** num_qubits, 1))
    zero[0, 0] = 1
    return qiskit.quantum_info.Statevector(zero)

def _random_qubit_unitary(num_qubits: int):
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(random_matrix)
    return qiskit.quantum_info.Operator(q)

def _random_qubit_state(num_qubits: int):
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return qiskit.quantum_info.Statevector(vec)

def random_training_data(unitary: qiskit.quantum_info.Operator, samples: int):
    dataset = []
    num_qubits = unitary.data.shape[0].bit_length() - 1
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    unitaries = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = op @ _tensored_id(num_outputs - 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def state_fidelity(a: qiskit.quantum_info.Statevector, b: qiskit.quantum_info.Statevector) -> float:
    return abs(np.vdot(a.data, b.data)) ** 2

def fidelity_adjacency(states: Sequence[qiskit.quantum_info.Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNGen:
    """
    Hybrid quantum graph neural network that can run on CPU or Qiskit Aer simulator.

    The class keeps the original feedforward semantics while adding:
    1. Automatic device selection (CPU or Aer simulator).
    2. A simple fidelity‑based regulariser on hidden state vectors.
    3. A lightweight training loop that optimises the unitary parameters by finite difference.
    """

    def __init__(self, arch: Sequence[int], device: str | None = None):
        self.arch = list(arch)
        self.device = device or ("aer_simulator" if qiskit.Aer.get_backend('aer_simulator').is_statevector() else "qasm_simulator")
        self.backend = Aer.get_backend(self.device)

    def feedforward(self, unitaries: Sequence[Sequence[qiskit.quantum_info.Operator]], samples: Iterable[Tuple[qiskit.quantum_info.Statevector, qiskit.quantum_info.Statevector]]):
        stored_states = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.arch)):
                unitary = unitaries[layer][0]
                for gate in unitaries[layer][1:]:
                    unitary = gate @ unitary
                current_state = unitary @ current_state
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    def fidelity_regulariser(self, activations):
        reg = 0.0
        for sample_acts in activations[1:]:
            for i in range(len(sample_acts) - 1):
                for j in range(i + 1, len(sample_acts)):
                    a = sample_acts[i]
                    b = sample_acts[j]
                    fid = state_fidelity(a, b)
                    reg += fid ** 2
        return reg

    def train(self, unitaries, samples, epochs=10, lr=0.01, reg_weight=0.0):
        params = []
        for layer in range(1, len(self.arch)):
            for op in unitaries[layer]:
                params.append(op.data.flatten())
        params = np.concatenate(params)
        for epoch in range(epochs):
            loss = 0.0
            grad = np.zeros_like(params)
            for sample, target in samples:
                # Forward pass
                layerwise = [sample]
                current_state = sample
                for layer in range(1, len(self.arch)):
                    unitary = unitaries[layer][0]
                    for gate in unitaries[layer][1:]:
                        unitary = gate @ unitary
                    current_state = unitary @ current_state
                    layerwise.append(current_state)
                pred = layerwise[-1]
                loss += np.linalg.norm(pred.data - target.data) ** 2
                # Simple finite‑difference gradient
                eps = 1e-5
                for idx in range(len(params)):
                    perturbed = params.copy()
                    perturbed[idx] += eps
                    # rebuild unitaries from perturbed params
                    offset = 0
                    for layer in range(1, len(self.arch)):
                        for op in unitaries[layer]:
                            shape = op.data.shape
                            size = shape[0] * shape[1]
                            new_data = perturbed[offset:offset + size].reshape(shape)
                            op.data = new_data
                            offset += size
                    # forward with perturbed params
                    pert_layerwise = [sample]
                    pert_current = sample
                    for layer in range(1, len(self.arch)):
                        unitary = unitaries[layer][0]
                        for gate in unitaries[layer][1:]:
                            unitary = gate @ unitary
                        pert_current = unitary @ pert_current
                        pert_layerwise.append(pert_current)
                    pert_pred = pert_layerwise[-1]
                    pert_loss = np.linalg.norm(pert_pred.data - target.data) ** 2
                    grad[idx] += (pert_loss - loss) / eps
            loss /= len(samples)
            if reg_weight > 0.0:
                reg = self.fidelity_regulariser(layerwise)
                loss += reg_weight * reg
            # Parameter update
            params -= lr * grad / len(samples)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
