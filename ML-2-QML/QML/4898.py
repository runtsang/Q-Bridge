"""Unified hybrid QNN in quantum mode.

This module mirrors the classical implementation but replaces the core
layers with quantum circuits and simulators.  The same class name
`HybridQNN` is used so that downstream code can switch between
``qiskit`` and ``qutip`` back‑ends without changing the API.

Typical usage:

```python
from UnifiedEstimatorQNN import HybridQNN

est = HybridQNN(mode='estimator', backend='qiskit')
conv = HybridQNN(mode='conv', backend='qiskit', kernel_size=3)
clf = HybridQNN(mode='classifier', backend='qiskit', num_qubits=5, depth=2)
graph = HybridQNN(mode='graph', backend='qutip', qnn_arch=[3,4,5], samples=200)
```

"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
import qiskit.circuit.random as qiskit_random
import qiskit_machine_learning.neural_networks as qml_nn
import qutip as qt
import scipy as sc

__all__ = ["HybridQNN", "EstimatorQNN", "Conv", "QuantumClassifier", "GraphQNN"]


# --------------------------------------------------------------------------- #
# 1. Quantum Estimator                                                    #
# --------------------------------------------------------------------------- #
def _build_qiskit_estimator() -> Tuple[qiskit.circuit.QuantumCircuit, List[qiskit.circuit.Parameter]]:
    """Return a minimal 1‑qubit variational circuit with two parameters."""
    params = [qiskit.circuit.Parameter(f"p{i}") for i in range(2)]
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    return qc, params


# --------------------------------------------------------------------------- #
# 2. Quantum convolution (quanvolution)                                   #
# --------------------------------------------------------------------------- #
class _QuanvCircuit:
    """Quantum filter that mimics the classical Conv filter."""

    def __init__(self, kernel_size: int, backend: qiskit.providers.Provider, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"θ{i}") for i in range(self.n_qubits)]
        # Encode pixel intensities
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit_random.random_circuit(self.n_qubits, depth=2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Return the mean probability of measuring |1> across qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


# --------------------------------------------------------------------------- #
# 3. Quantum classifier                                                   #
# --------------------------------------------------------------------------- #
def _build_qiskit_classifier(num_qubits: int, depth: int) -> Tuple[qiskit.circuit.QuantumCircuit, List[qiskit.circuit.Parameter], List[qiskit.circuit.Parameter], List[qt.Qobj]]:
    """Return a layered ansatz with explicit data encoding and variational weights."""
    encoding = qiskit.circuit.ParameterVector("x", num_qubits)
    weights = qiskit.circuit.ParameterVector("θ", num_qubits * depth)
    qc = qiskit.QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)
    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)
    # Observables (Z on each qubit)
    obs = [qt.Qobj.from_label("Z" + "I" * (num_qubits - 1))]
    return qc, list(encoding), list(weights), obs


# --------------------------------------------------------------------------- #
# 4. Graph‑based QNN utilities (QuTiP)                                    #
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator with explicit qubit dimensions."""
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector with explicit qubit dimensions."""
    proj = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    proj.dims = [dims.copy(), dims.copy()]
    return proj


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    U = sc.linalg.orth(mat)
    qobj = qt.Qobj(U)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amp = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amp /= sc.linalg.norm(amp)
    state = qt.Qobj(amp)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def _random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    return [( _random_qubit_state(len(unitary.dims[0])), unitary * _random_qubit_state(len(unitary.dims[0])) ) for _ in range(samples)]


def _random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = _random_training_data(target_unitary, samples)
    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary


def _feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Forward propagate a list of pure states through the QNN."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        current = sample
        layerwise = [current]
        for layer in range(1, len(qnn_arch)):
            # layer unitary composed of all gates in the layer
            unitary = unitaries[layer][0]
            for gate in unitaries[layer][1:]:
                unitary = gate * unitary
            # apply unitary and trace out unnecessary qubits
            state = qt.tensor(current, _tensored_zero(num_outputs := qnn_arch[layer]))
            state = unitary * state * unitary.dag()
            keep = list(range(num_outputs))
            state = state.ptrace(keep)
            current = state
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


def _state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def _fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                        *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 5. Unified class                                                            #
# --------------------------------------------------------------------------- #
class HybridQNN:
    """Unified estimator that can operate in four distinct modes.

    The quantum back‑end uses Qiskit for the estimator, conv, and
    classifier, while QuTiP is used for graph‑based utilities.
    """

    def __init__(self, mode: str, backend: str = 'qiskit', **kwargs):
        self.mode = mode
        self.backend = backend

        if mode == 'estimator':
            self.circuit, self.params = _build_qiskit_estimator()
            self.estimator = qiskit.primitives.StatevectorEstimator()
            self.estimator_qnn = qml_nn.EstimatorQNN(
                circuit=self.circuit,
                observables=[qt.Qobj.from_label("Y")],
                input_params=[self.params[0]],
                weight_params=[self.params[1]],
                estimator=self.estimator,
            )

        elif mode == 'conv':
            kernel_size = kwargs.get('kernel_size', 2)
            self.filter = _QuanvCircuit(kernel_size=kernel_size,
                                        backend=qiskit.Aer.get_backend("qasm_simulator"),
                                        shots=1024,
                                        threshold=0.5)

        elif mode == 'classifier':
            num_qubits = kwargs.get('num_qubits', 5)
            depth = kwargs.get('depth', 2)
            self.circuit, self.encoding, self.weights, self.observables = _build_qiskit_classifier(num_qubits, depth)
            self.estimator = qiskit.primitives.Estimator()
            self.estimator_qnn = qml_nn.EstimatorQNN(
                circuit=self.circuit,
                observables=self.observables,
                input_params=self.encoding,
                weight_params=self.weights,
                estimator=self.estimator,
            )

        elif mode == 'graph':
            qnn_arch = kwargs.get('qnn_arch', [3,4,5])
            samples = kwargs.get('samples', 200)
            self.arch, self.unitaries, self.training_data, self.target_unitary = _random_network(qnn_arch, samples)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def run(self, inputs):
        """Execute the quantum routine for the chosen mode."""
        if self.mode == 'estimator':
            return self.estimator_qnn.run(inputs)
        elif self.mode == 'conv':
            return self.filter.run(inputs)
        elif self.mode == 'classifier':
            return self.estimator_qnn.run(inputs)
        else:
            raise NotImplementedError("Run not defined for graph mode.")

    # --------------------------------------------------------------------- #
    # Graph utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        return _random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        return _feedforward(qnn_arch, unitaries, samples)

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return _fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


# --------------------------------------------------------------------------- #
# Convenience wrappers for backward compatibility                           #
# --------------------------------------------------------------------------- #
def EstimatorQNN(**kwargs):
    """Return a HybridQNN instance in estimator mode."""
    return HybridQNN(mode='estimator', **kwargs)

def Conv(**kwargs):
    """Return a HybridQNN instance in conv mode."""
    return HybridQNN(mode='conv', **kwargs)

def QuantumClassifier(**kwargs):
    """Return a HybridQNN instance in classifier mode."""
    return HybridQNN(mode='classifier', **kwargs)

def GraphQNN(**kwargs):
    """Return a HybridQNN instance in graph mode."""
    return HybridQNN(mode='graph', **kwargs)
