from __future__ import annotations

import itertools
from typing import List, Sequence

import qiskit
import numpy as np
import scipy as sc
import qutip as qt
import networkx as nx


class QuantumNATGen279:
    """
    Quantum implementation of the hybrid architecture.  The model builds a
    parameterised Qiskit circuit that mirrors the classical CNN encoder,
    a random unitary layer, and a fully‑connected readout.  It also
    provides fidelity‑based graph construction using qutip states.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.circuit = self._build_circuit()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        # Encode classical features via Ry rotations
        for i in range(self.n_qubits):
            qc.ry(qiskit.circuit.Parameter(f"theta_{i}"), i)
        # Random unitary layer
        unitary = self._random_unitary(self.n_qubits)
        qc.unitary(unitary, list(range(self.n_qubits)), label="U")
        # Measurement
        qc.measure_all()
        return qc

    def _random_unitary(self, n: int) -> np.ndarray:
        dim = 2 ** n
        mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
        q, _ = sc.linalg.qr(mat)
        return q

    def run(self, params: List[float]) -> np.ndarray:
        bound = {f"theta_{i}": p for i, p in enumerate(params)}
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bound],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array([counts.get(state, 0) for state in self._state_bin_iter()]) / self.shots
        expectation = np.sum(probs * np.array([self._state_value(state) for state in self._state_bin_iter()]))
        return np.array([expectation])

    def _state_bin_iter(self):
        return [format(i, f"0{self.n_qubits}b") for i in range(2 ** self.n_qubits)]

    def _state_value(self, state: str) -> float:
        # Map bitstring to a value in [-1, 1] for demonstration
        return 1.0 if state.count("1") % 2 == 0 else -1.0

    # ------------------------------------------------------------------
    # Fidelity graph utilities using qutip
    # ------------------------------------------------------------------
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
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return state.ptrace(keep)

    @staticmethod
    def _state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: List[qt.Qobj],
        threshold: float,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["QuantumNATGen279"]
