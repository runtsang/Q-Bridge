"""UnifiedSelfAttentionQNN – quantum component.

The quantum module implements a variational self‑attention circuit that
* uses a rotation‑parameterised RX/RY/RZ block for each qubit,
* entangles adjacent qubits with controlled‑RX gates,
* measures in the computational basis and returns the measurement
  histogram as a probability distribution over the attention heads.
The output is then fed into a quantum‑graph neural network (QGN) that
propagates a quantum state through a sequence of random unitary layers.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, AerSimulator
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import CX, RX, RY, RZ, CRX

import qutip as qt
import scipy as sc
import networkx as nx
from typing import Sequence, Tuple, List, Iterable
from collections.abc import Iterable as IterableABC


class UnifiedSelfAttentionQNN:
    """
    Quantum‑classical hybrid architecture that mirrors the classical
    attention mechanism but entirely in the quantum domain.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the self‑attention block.
    qnn_arch : Sequence[int]
        Layer sizes for the graph‑structured quantum neural network.
    """

    def __init__(self, n_qubits: int, qnn_arch: Sequence[int]):
        self.n_qubits = n_qubits
        self.qnn_arch = list(qnn_arch)

        # Quantum simulator
        self.sim = AerSimulator()

        # Build self‑attention circuit template
        self.attn_circuit, self.attn_params = self._build_attention_circuit()

        # Random QNN unitaries
        self.qnn_unitaries = self._random_qnn_unitaries()

    # --------------------------------------------------------------------------- #
    # Self‑attention circuit construction
    # --------------------------------------------------------------------------- #
    def _build_attention_circuit(self) -> Tuple[QuantumCircuit, List[Parameter]]:
        """
        Create a parameterised circuit that implements
        RX/RY/RZ rotations on each qubit followed by a chain of
        controlled‑RX entangling gates.
        """
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        params = []

        # Rotation parameters
        for i in range(self.n_qubits):
            rx_p = Parameter(f'θ_{i}_rx')
            ry_p = Parameter(f'θ_{i}_ry')
            rz_p = Parameter(f'θ_{i}_rz')
            params.extend([rx_p, ry_p, rz_p])
            qc.rx(rx_p, i)
            qc.ry(ry_p, i)
            qc.rz(rz_p, i)

        # Entangling parameters
        for i in range(self.n_qubits - 1):
            crx_p = Parameter(f'φ_{i}')
            params.append(crx_p)
            qc.crx(crx_p, i, i + 1)

        qc.measure_all()
        return qc, params

    # --------------------------------------------------------------------------- #
    # Run self‑attention
    # --------------------------------------------------------------------------- #
    def run_attention(self,
                      rotation_params: np.ndarray,
                      entangle_params: np.ndarray,
                      shots: int = 1024) -> dict:
        """
        Bind the supplied parameters, execute the circuit and return
        the measurement histogram.
        """
        if len(rotation_params)!= 3 * self.n_qubits:
            raise ValueError("rotation_params must have length 3 * n_qubits")
        if len(entangle_params)!= self.n_qubits - 1:
            raise ValueError("entangle_params must have length n_qubits - 1")

        # Map parameters to circuit
        param_dict = {}
        idx = 0
        for i in range(self.n_qubits):
            param_dict[self.attn_params[idx]] = rotation_params[3 * i]
            param_dict[self.attn_params[idx + 1]] = rotation_params[3 * i + 1]
            param_dict[self.attn_params[idx + 2]] = rotation_params[3 * i + 2]
            idx += 3
        for i in range(self.n_qubits - 1):
            param_dict[self.attn_params[idx]] = entangle_params[i]
            idx += 1

        bound_qc = self.attn_circuit.bind_parameters(param_dict)
        job = self.sim.run(bound_qc, shots=shots)
        return job.result().get_counts(bound_qc)

    # --------------------------------------------------------------------------- #
    # Quantum Graph Neural Network utilities
    # --------------------------------------------------------------------------- #
    def _random_qubit_unitary(self, num_qubits: int) -> qt.Qobj:
        """
        Generate a random unitary on `num_qubits` qubits.
        """
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])

    def _swap_registers(self, op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        """
        Permute the register order of a Qobj.
        """
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    def _random_qnn_unitaries(self) -> List[List[qt.Qobj]]:
        """
        Construct a list of unitary layers for the quantum graph neural network.
        Each layer contains `num_outputs` unitaries acting on
        `num_inputs + 1` qubits (the +1 accounts for a dummy qubit that
        is later traced out).
        """
        unitaries = []
        for layer in range(1, len(self.qnn_arch)):
            num_inputs = self.qnn_arch[layer - 1]
            num_outputs = self.qnn_arch[layer]
            layer_ops = []
            for out in range(num_outputs):
                op = self._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(op, qt.qeye(2 ** (num_outputs - 1)))
                    op = self._swap_registers(op, num_inputs, num_inputs + out)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return unitaries

    # --------------------------------------------------------------------------- #
    # Run QNN forward pass
    # --------------------------------------------------------------------------- #
    def run_qnn(self,
                input_state: qt.Qobj,
                shots: int = 1024) -> dict:
        """
        Propagate an input quantum state through the QNN and return
        a measurement histogram of the final state.
        """
        state = input_state
        for layer_ops in self.qnn_unitaries:
            # For simplicity, apply the first unitary and trace out the extra qubit
            unitary = layer_ops[0]
            state = unitary * state * unitary.dag()
            # Partial trace out the first qubit (index 0)
            state = state.ptrace([i for i in range(state.dims[0].size() - 1)])

        # Convert final state to probabilities
        probs = state.probabilities()
        counts = {}
        for i, p in enumerate(probs):
            bits = format(i, f'0{state.dims[0].size() - 1}b')
            counts[bits] = int(p * shots)
        return counts

    # --------------------------------------------------------------------------- #
    # Combined run
    # --------------------------------------------------------------------------- #
    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            input_state: qt.Qobj,
            shots: int = 1024) -> Tuple[dict, dict]:
        """
        Execute both the self‑attention circuit and the QNN.
        Returns a tuple of measurement histograms.
        """
        attn_counts = self.run_attention(rotation_params, entangle_params, shots)
        qnn_counts = self.run_qnn(input_state, shots)
        return attn_counts, qnn_counts

    # --------------------------------------------------------------------------- #
    # Static helpers (mirroring the classical GraphQNN utilities)
    # --------------------------------------------------------------------------- #
    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """
        Generate random training pairs (state, unitary * state).
        """
        dataset = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = UnifiedSelfAttentionQNN._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amps /= sc.linalg.norm(amps)
        state = qt.Qobj(amps)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        """
        Construct a random QNN architecture and corresponding training data.
        """
        target_unitary = UnifiedSelfAttentionQNN._random_qubit_unitary_static(qnn_arch[-1])
        training_data = UnifiedSelfAttentionQNN.random_training_data(target_unitary, samples)

        unitaries = []
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops = []
            for out in range(num_outputs):
                op = UnifiedSelfAttentionQNN._random_qubit_unitary_static(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(op, qt.qeye(2 ** (num_outputs - 1)))
                    op = UnifiedSelfAttentionQNN._swap_registers_static(op, num_inputs, num_inputs + out)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def _random_qubit_unitary_static(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        unitary = sc.linalg.orth(matrix)
        return qt.Qobj(unitary, dims=[[2] * num_qubits, [2] * num_qubits])

    @staticmethod
    def _swap_registers_static(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """
        Return the absolute squared overlap between pure states ``a`` and ``b``.
        """
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[qt.Qobj],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """
        Create a weighted adjacency graph from state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in zip(enumerate(states), enumerate(states[1:])):
            fid = UnifiedSelfAttentionQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["UnifiedSelfAttentionQNN"]
