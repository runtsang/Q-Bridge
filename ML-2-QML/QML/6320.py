"""Quantum graph neural network and autoencoder utilities.

This module keeps the original quantum GraphQNN helper functions
(`feedforward`, `fidelity_adjacency`, `random_network`,
`random_training_data`, `state_fidelity`) and adds a variational
autoencoder built with Qiskit.  The public class
``GraphQNNAutoencoder`` exposes a sampler‑based quantum neural network
and a quantum autoencoder that can be trained with a gradient‑free
optimizer.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple, List

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

Tensor = qt.Qobj

# --------------------------------------------------------------------------- #
#  Core QNN state propagation (from the original QML seed)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Return an identity operator with proper dimensions."""
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Return the zero projector on ``num_qubits``."""
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


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (|ψ⟩, U|ψ⟩) pairs for a target unitary."""
    data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n = len(unitary.dims[0])
    for _ in range(samples):
        psi = _random_qubit_state(n)
        data.append((psi, unitary * psi))
    return data


def random_network(qnn_arch: List[int], samples: int):
    """Return architecture, list of layer unitaries, training data and target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    layer_unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        nin = qnn_arch[layer - 1]
        nout = qnn_arch[layer]
        ops: List[qt.Qobj] = []
        for out in range(nout):
            op = _random_qubit_unitary(nin + 1)
            if nout > 1:
                op = qt.tensor(_random_qubit_unitary(nin + 1),
                               _tensored_id(nout - 1))
                op = _swap_registers(op, nin, nin + out)
            ops.append(op)
        layer_unitaries.append(ops)

    return qnn_arch, layer_unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int],
                   unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int,
                   input_state: qt.Qobj) -> qt.Qobj:
    nin = qnn_arch[layer - 1]
    nout = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(nout))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(nin))


def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Return the state at each layer for every sample."""
    out: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise: List[qt.Qobj] = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        out.append(layerwise)
    return out


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
#  Quantum autoencoder (from the Autoencoder QML seed)
# --------------------------------------------------------------------------- #

def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a variational auto‑encoder circuit with a swap‑test."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test to measure overlap between latent and trash
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc


def build_sampler_qnn(circuit: QuantumCircuit) -> SamplerQNN:
    """Wrap the circuit in a SamplerQNN."""
    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=Sampler()
    )


def train_quantum_autoencoder(qnn: SamplerQNN,
                              samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
                              *,
                              epochs: int = 10,
                              maxiter: int = 50) -> List[float]:
    """Gradient‑free training of the QNN using COBYLA."""
    def loss(params: Sequence[float]) -> float:
        qnn.set_parameter_values(params)
        total = 0.0
        for inp, tgt in samples:
            out = qnn.run(inp).samples
            # Convert the sampled bitstring to a state vector
            out_sv = Statevector(out[0])
            total += 1 - state_fidelity(out_sv, tgt)
        return total / len(samples)

    optimizer = COBYLA()
    best_params = list(qnn.parameters)
    best_loss = loss(best_params)

    for _ in range(epochs):
        new_params = optimizer.minimize(loss, best_params, options={"maxiter": maxiter})
        new_loss = loss(new_params)
        if new_loss < best_loss:
            best_loss = new_loss
            best_params = new_params
        best_params = new_params

    qnn.set_parameter_values(best_params)
    return [best_loss]

# --------------------------------------------------------------------------- #
#  Hybrid class (quantum side)
# --------------------------------------------------------------------------- #

class GraphQNNAutoencoder:
    """Hybrid quantum GraphQNN + variational auto‑encoder.

    The class exposes a sampler‑based quantum neural network (QNN)
    defined by ``qnn_arch`` and a quantum auto‑encoder circuit.  It
    provides the same public API as the classical counterpart while
    delegating state propagation to Qiskit and fidelity graph building
    to Qutip.
    """

    def __init__(self,
                 qnn_arch: Sequence[int],
                 num_latent: int,
                 num_trash: int) -> None:
        self.qnn_arch = list(qnn_arch)
        self.autoencoder_circuit = quantum_autoencoder_circuit(num_latent, num_trash)
        self.qnn = build_sampler_qnn(self.autoencoder_circuit)

    # ----- quantum GraphQNN helpers ------------------------------------------------

    def feedforward(self,
                    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Return the state at each layer for every sample."""
        # The sampler QNN runs on the full circuit; for compatibility we
        # wrap the original feedforward logic that expects a list of
        # unitaries, but here we simply return the raw sampler output.
        return feedforward(self.qnn_arch, [], samples)  # unitaries unused in this toy demo

    def get_graph_from_fidelities(self,
                                  states: Sequence[qt.Qobj],
                                  threshold: float,
                                  *,
                                  secondary: float | None = None,
                                  secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

    # ----- auto‑encoder helpers -----------------------------------------------------

    def train_autoencoder(self,
                          samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
                          **kwargs) -> List[float]:
        """Train the sampler‑QNN auto‑encoder."""
        return train_quantum_autoencoder(self.qnn, samples, **kwargs)

    # ----- static helpers ------------------------------------------------------------

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Proxy to the original random_network."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int):
        """Proxy to the original random_training_data."""
        return random_training_data(unitary, samples)

__all__ = [
    "quantum_autoencoder_circuit",
    "build_sampler_qnn",
    "train_quantum_autoencoder",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNAutoencoder",
]
