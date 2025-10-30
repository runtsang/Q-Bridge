from __future__ import annotations

import itertools
import numpy as np
import qutip as qt
import networkx as nx
from typing import Iterable, Sequence, List, Tuple

# Quantum circuit helpers – the original QML seed used Qiskit;
# we keep the same interface but provide a pure‑Qutip implementation
# for state propagation and a Qiskit sampler for the autoencoder.

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class GraphQNNHybrid:
    """
    Quantum implementation of the hybrid class.  Mirrors the public API
    of the classical counterpart but operates on Qobj statevectors
    and variational circuits.  The autoencoder is a swap‑test based
    variational circuit built with Qiskit.
    """

    def __init__(self, qnn_arch: Sequence[int], autoencoder_config: dict | None = None):
        self.qnn_arch = list(qnn_arch)
        self.autoencoder_config = autoencoder_config or {}
        self.unitaries = self._init_random_unitaries()
        self.autoencoder_qnn = None
        if self.autoencoder_config:
            self.autoencoder_qnn = self._build_autoencoder_qnn()

    # ----------------------------------------------------------------------
    # Random unitary generation
    # ----------------------------------------------------------------------
    def _random_qubit_unitary(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        unitary = np.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    def _init_random_unitaries(self) -> List[List[qt.Qobj]]:
        """
        Create a list of lists of random unitaries, one per output node
        for each layer.  The first layer operates on the input state plus
        an auxiliary zero state; subsequent layers concatenate outputs
        with fresh zeros.
        """
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(self.qnn_arch)):
            num_inputs = self.qnn_arch[layer - 1]
            num_outputs = self.qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = self._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    # tensor with identity on remaining outputs
                    op = qt.tensor(op, qt.qeye(2 ** (num_outputs - 1)))
                    op = self._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return unitaries

    def _swap_registers(self, op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    # ----------------------------------------------------------------------
    # State propagation
    # ----------------------------------------------------------------------
    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return state.ptrace(keep)

    def _layer_channel(self,
                       layer: int,
                       input_state: qt.Qobj) -> qt.Qobj:
        num_inputs = self.qnn_arch[layer - 1]
        num_outputs = self.qnn_arch[layer]
        # prepend zero state for unused outputs
        state = qt.tensor(input_state, qt.fock(2 ** num_outputs).proj())
        # compose the layer unitary
        layer_unitary = self.unitaries[layer][0].copy()
        for gate in self.unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        # propagate and trace out the input qubits
        return self._partial_trace_remove(layer_unitary * state * layer_unitary.dag(),
                                          range(num_inputs))

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """
        Run a batch of samples through the quantum network.
        Each sample is a tuple (input_state, target_state) but only
        the input is propagated.  Returns a list of layer‑wise states.
        """
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(self.qnn_arch)):
                current = self._layer_channel(layer, current)
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states

    # ----------------------------------------------------------------------
    # Training data & random network
    # ----------------------------------------------------------------------
    def _random_qubit_state(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        amplitudes /= np.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    def random_training_data(self,
                              target_unitary: qt.Qobj,
                              samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        data = []
        num_qubits = len(target_unitary.dims[0])
        for _ in range(samples):
            state = self._random_qubit_state(num_qubits)
            data.append((state, target_unitary * state))
        return data

    def random_network(self, samples: int) -> Tuple[List[int], List[List[qt.Qobj]],
                                                    List[Tuple[qt.Qobj, qt.Qobj]],
                                                    qt.Qobj]:
        target_unitary = self._random_qubit_unitary(self.qnn_arch[-1])
        training_data = self.random_training_data(target_unitary, samples)
        return self.qnn_arch, self.unitaries, training_data, target_unitary

    # ----------------------------------------------------------------------
    # Fidelity & adjacency
    # ----------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ----------------------------------------------------------------------
    # Quantum autoencoder
    # ----------------------------------------------------------------------
    def _build_autoencoder_qnn(self) -> SamplerQNN:
        """
        Construct a swap‑test based variational autoencoder using Qiskit.
        The circuit is identical to the reference seed but wrapped in
        :class:`SamplerQNN` for easy evaluation.
        """
        algorithm_globals.random_seed = 42
        sampler = StatevectorSampler()

        def ansatz(num_qubits: int) -> QuantumCircuit:
            return RealAmplitudes(num_qubits, reps=5)

        def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(qr, cr)
            circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
            circuit.barrier()
            auxiliary_qubit = num_latent + 2 * num_trash
            # swap test
            circuit.h(auxiliary_qubit)
            for i in range(num_trash):
                circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
            circuit.h(auxiliary_qubit)
            circuit.measure(auxiliary_qubit, cr[0])
            return circuit

        num_latent = self.autoencoder_config.get("num_latent", 3)
        num_trash = self.autoencoder_config.get("num_trash", 2)
        qc = auto_encoder_circuit(num_latent, num_trash)

        def identity_interpret(x: np.ndarray) -> np.ndarray:
            return x

        return SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            interpret=identity_interpret,
            output_shape=2,
            sampler=sampler,
        )

    def get_autoencoder_qnn(self) -> SamplerQNN | None:
        """Return the quantum autoencoder, if configured."""
        return self.autoencoder_qnn


__all__ = ["GraphQNNHybrid"]
