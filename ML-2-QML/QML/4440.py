"""Hybrid quantum layer mirroring the classical FCL interface.

The module supports four back‑ends:
* qiskit – fully‑connected or convolutional circuits
* strawberryfields – photonic fraud‑detection program
* qutip – graph‑based state propagation
* networkx – fidelity graph construction
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import numpy as np
import networkx as nx
import qiskit
import qutip as qt
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

Tensor = qt.Qobj


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FCL:
    """
    Quantum hybrid layer.

    Parameters
    ----------
    mode : {'qiskit', 'convolution', 'fraud', 'graph'}
        The underlying quantum implementation.
    **kwargs : dict
        Mode‑specific keyword arguments.
    """

    def __init__(self, mode: str = "qiskit", **kwargs) -> None:
        self.mode = mode

        if mode == "qiskit":
            n_qubits = kwargs.get("n_qubits", 1)
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = 100
            self.circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self.circuit.h(range(n_qubits))
            self.circuit.barrier()
            self.circuit.ry(self.theta, range(n_qubits))
            self.circuit.measure_all()
        elif mode == "convolution":
            kernel_size = kwargs.get("kernel_size", 2)
            threshold = kwargs.get("threshold", 127)
            self.n_qubits = kernel_size ** 2
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = 100
            self.circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self.circuit.rx(self.theta[i], i)
            self.circuit.barrier()
            self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
            self.circuit.measure_all()
            self.threshold = threshold
        elif mode == "fraud":
            self.program = self._build_fraud_program(kwargs["params"])
            self.engine = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        elif mode == "graph":
            self.arch: List[int] = kwargs["arch"]
            self.unitary = self._random_qubit_unitary(self.arch[-1])
            self.training_data = self._random_training_data(self.unitary, 10)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _build_fraud_program(self, params: FraudLayerParameters) -> sf.Program:
        program = sf.Program(2)
        with program.context as q:
            self._apply_layer(q, params, clip=False)
        return program

    def _apply_layer(self, modes, params: FraudLayerParameters, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _random_qubit_unitary(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(
            size=(dim, dim)
        )
        unitary, _ = np.linalg.qr(matrix)
        return qt.Qobj(unitary)

    def _random_qubit_state(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amps = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(
            size=(dim, 1)
        )
        amps /= np.linalg.norm(amps)
        return qt.Qobj(amps)

    def _random_training_data(self, unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        data = []
        for _ in range(samples):
            state = self._random_qubit_state(len(unitary.dims[0]))
            data.append((state, unitary * state))
        return data

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        if self.mode == "qiskit":
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probs = counts / self.shots
            expectation = np.sum(states * probs)
            return np.array([expectation])
        if self.mode == "convolution":
            param_binds = []
            for dat in thetas:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self.circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)
        if self.mode == "fraud":
            result = self.engine.run(self.program, args={})
            return result.samples
        if self.mode == "graph":
            outputs = []
            for state, _ in self.training_data:
                current = state
                for _ in self.arch[1:]:
                    current = self._layer_channel(current)
                outputs.append(current)
            return outputs
        raise RuntimeError("unreachable")

    def _layer_channel(self, state: qt.Qobj) -> qt.Qobj:
        unitary = self._random_qubit_unitary(len(state.dims[0]))
        return unitary * state * unitary.dag()


# --------------------------------------------------------------------------- #
# Graph‑QNN utilities – quantum version
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
) -> List[List[qt.Qobj]]:
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
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


__all__ = [
    "FCL",
    "FraudLayerParameters",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
