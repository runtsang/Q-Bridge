import itertools
import numpy as np
import qiskit
import qutip as qt
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

class GraphQNN:
    """Quantum graph neural network mirroring the classical API.

    Features:
        * Random unitary layers generated from Haar measure.
        * Variational feed‑forward propagation on Qobj states.
        * Fidelity‑based adjacency graph construction.
        * A parameterised quantum circuit serving as a fully‑connected
          layer analogue (FCL) implemented with Qiskit.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)

    # ------------------- Helper primitives -------------------------
    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        I = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        I.dims = [dims.copy(), dims.copy()]
        return I

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        Z = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        Z.dims = [dims.copy(), dims.copy()]
        return Z

    @staticmethod
    def _swap_registers(op: qt.Qobj, src: int, tgt: int) -> qt.Qobj:
        if src == tgt:
            return op
        order = list(range(len(op.dims[0])))
        order[src], order[tgt] = order[tgt], order[src]
        return op.permute(order)

    @staticmethod
    def _random_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        return qt.Qobj(q)

    @staticmethod
    def _random_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        return qt.Qobj(vec)

    # ------------------- Random network generation ------------------
    def random_network(self, samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        target = self._random_unitary(self.arch[-1])
        train = [(self._random_state(len(target.dims[0])),
                  target * self._random_state(len(target.dims[0]))) for _ in range(samples)]

        layers: List[List[qt.Qobj]] = [[]]
        for l in range(1, len(self.arch)):
            in_q = self.arch[l - 1]
            out_q = self.arch[l]
            ops: List[qt.Qobj] = []
            for out in range(out_q):
                op = self._random_unitary(in_q + 1)
                if out_q > 1:
                    op = qt.tensor(self._random_unitary(in_q + 1), self._tensored_id(out_q - 1))
                    op = self._swap_registers(op, in_q, in_q + out)
                ops.append(op)
            layers.append(ops)
        return self.arch, layers, train, target

    # ------------------- Forward propagation ------------------------
    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for idx in sorted(remove, reverse=True):
            keep.pop(idx)
        return state.ptrace(keep)

    def _layer_channel(self, layer: int, unitaries: List[qt.Qobj], inp: qt.Qobj) -> qt.Qobj:
        num_in = self.arch[layer - 1]
        num_out = self.arch[layer]
        state = qt.tensor(inp, self._tensored_zero(num_out))
        op = unitaries[0]
        for gate in unitaries[1:]:
            op = gate * op
        return self._partial_trace_remove(op * state * op.dag(), range(num_in))

    def feedforward(self, unitaries: Iterable[List[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        out = []
        for inp, _ in samples:
            states = [inp]
            cur = inp
            for l, ops in enumerate(unitaries[1:], start=1):
                cur = self._layer_channel(l, ops, cur)
                states.append(cur)
            out.append(states)
        return out

    # ------------------- Fidelity helpers --------------------------
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
            fid = GraphQNN.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------- Quantum FCL --------------------------------
    def FCL(self):
        """Return a Qiskit circuit mimicking the classical FCL."""
        class _QuantumFCL:
            def __init__(self, n_qubits: int = 1, shots: int = 100):
                self.circuit = qiskit.QuantumCircuit(n_qubits)
                self.theta = qiskit.circuit.Parameter("theta")
                self.circuit.h(range(n_qubits))
                self.circuit.barrier()
                self.circuit.ry(self.theta, range(n_qubits))
                self.circuit.measure_all()
                self.backend = qiskit.Aer.get_backend("qasm_simulator")
                self.shots = shots

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                job = qiskit.execute(
                    self.circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[{self.theta: t} for t in thetas],
                )
                result = job.result().get_counts(self.circuit)
                counts = np.array(list(result.values()))
                states = np.array(list(result.keys())).astype(float)
                probs = counts / self.shots
                return np.array([np.sum(states * probs)])

        return _QuantumFCL()

__all__ = [
    "GraphQNN",
]
