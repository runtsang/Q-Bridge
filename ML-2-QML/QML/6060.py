"""GraphQNN__gen480: Quantum graph neural network with parameterized variational circuits."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qutip as qt
import pennylane as qml
import pennylane.numpy as pnp

State = qt.Qobj
Tensor = qt.Qobj


class GraphQNN__gen480:
    """Quantum graph neural network that extends the original design by
    introducing a parameter‑shaped variational circuit per layer.
    The class provides methods to generate random networks, synthetic training
    data, feed‑forward propagation and fidelity‑based graph construction.

    Parameters
    ----------
    arch : Sequence[int]
        Number of qubits per layer, e.g. ``[2, 3, 3]`` creates a 2‑qubit input,
        a 3‑qubit hidden layer and a 3‑qubit output.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.num_layers = len(arch) - 1
        self.params: List[np.ndarray] = []

    # -------------------------------------------------------------------------
    # Random network generator
    # -------------------------------------------------------------------------
    @staticmethod
    def _random_unitary(num_qubits: int) -> qt.Qobj:
        """Return a Haar‑random unitary for *num_qubits* qubits."""
        dim = 2 ** num_qubits
        matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, r = np.linalg.qr(matrix)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        unitary = q @ np.diag(ph)
        return qt.Qobj(unitary)

    @staticmethod
    def _tensor_identity(num_qubits: int) -> qt.Qobj:
        return qt.tensor([qt.qeye(2) for _ in range(num_qubits)])

    @classmethod
    def random_network(
        cls,
        arch: Sequence[int],
        samples: int,
        *,
        seed: int | None = None,
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """
        Generate a random quantum network and a synthetic training set.

        Returns
        -------
        arch,
        unitaries,
        training_data,
        target_unitary
        """
        rng = np.random.default_rng(seed)
        target_unitary = cls._random_unitary(arch[-1])
        training_data = cls.random_training_data(target_unitary, samples, rng=rng)

        # Each layer is a list of gates acting on the qubits of the layer
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            in_q, out_q = arch[layer - 1], arch[layer]
            layer_gates: List[qt.Qobj] = []
            for _ in range(out_q):
                gate = cls._random_unitary(in_q + 1)  # +1 for ancilla
                layer_gates.append(gate)
            unitaries.append(layer_gates)

        return list(arch), unitaries, training_data, target_unitary

    @staticmethod
    def random_training_data(
        unitary: qt.Qobj, samples: int, *, rng: np.random.Generator | None = None
    ) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """
        Generate a dataset of input states and their images under *unitary*.
        """
        rng = rng or np.random.default_rng()
        dim = int(unitary.shape[0])
        data: List[Tuple[qt.Qobj, qt.Qobj]] = []
        for _ in range(samples):
            vec = rng.standard_normal((dim, 1)) + 1j * rng.standard_normal((dim, 1))
            vec /= np.linalg.norm(vec)
            state = qt.Qobj(vec)
            state.dims = [[2] * int(np.log2(dim)), [1] * int(np.log2(dim))]
            data.append((state, unitary * state))
        return data

    # -------------------------------------------------------------------------
    # Forward propagation
    # -------------------------------------------------------------------------
    def _layer_channel(
        self,
        layer: int,
        input_state: qt.Qobj,
        gates: List[qt.Qobj],
    ) -> qt.Qobj:
        """
        Apply a layer of gates to *input_state* and trace out the input qubits.
        """
        # Prepare ancilla qubits in |0>
        ancilla = self._tensor_identity(len(gates))
        full_state = qt.tensor(input_state, ancilla)

        # Compose the unitary for the entire layer
        unitary = gates[0]
        for g in gates[1:]:
            unitary = g * unitary

        # Apply and trace out the input qubits
        evolved = unitary * full_state * unitary.dag()
        in_q = len(input_state.dims[0])
        keep = list(range(in_q, in_q + len(gates)))  # keep output qubits
        return evolved.ptrace(keep)

    def feedforward(
        self,
        unitaries: List[List[qt.Qobj]],
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        """
        Propagate each input state through the network and return a list of
        state sequences for every example.
        """
        stored: List[List[qt.Qobj]] = []
        for inp, _ in samples:
            layerwise: List[qt.Qobj] = [inp]
            current = inp
            for layer_idx in range(1, len(self.arch)):
                current = self._layer_channel(layer_idx, current, unitaries[layer_idx])
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    # -------------------------------------------------------------------------
    # Fidelity helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Overlap squared between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen480.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # -------------------------------------------------------------------------
    # Variational circuit helpers (optional)
    # -------------------------------------------------------------------------
    def variational_parameters(self, seed: int | None = None) -> List[np.ndarray]:
        """
        Generate a list of parameter arrays, one per layer, suitable for a
        Pennylane variational circuit.
        """
        rng = np.random.default_rng(seed)
        params: List[np.ndarray] = []
        for layer in range(1, len(self.arch)):
            n_qubits = self.arch[layer]
            # One rotation per qubit per layer
            params.append(rng.standard_normal((n_qubits, 3)))
        self.params = params
        return params

    def variational_circuit(
        self, dev: qml.Device, params: List[np.ndarray]
    ) -> qml.QNode:
        """
        Build a Pennylane QNode that implements the variational circuit
        defined by *params*.
        """

        @qml.qnode(dev, interface="autograd")
        def circuit():
            for layer, (n_qubits, rot_params) in enumerate(zip(self.arch[1:], params)):
                for q in range(n_qubits):
                    qml.Rot(*rot_params[q], wires=q)
                if layer < len(self.arch) - 2:
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            return [qml.state()]

        return circuit
