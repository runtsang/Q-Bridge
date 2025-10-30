import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
import networkx as nx
import itertools

class ConvGraphQNN:
    """Hybrid quantum‑inspired convolution + graph neural network.

    The class mirrors the classical ``ConvGraphQNN`` but replaces the
    convolution with a parameterised quantum circuit and builds a
    fidelity graph from the resulting quantum states.  It is fully
    compatible with the original QML interface.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square quantum filter (number of qubits = k²).
    shots : int, optional
        Number of shots for the simulator.
    threshold : float, optional
        Classical threshold used to decide the rotation angle.
    graph_threshold : float, optional
        Fidelity threshold for graph edges.
    graph_secondary : float, optional
        Lower threshold for secondary weighted edges.
    graph_secondary_weight : float, optional
        Weight of secondary edges.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 shots: int = 100,
                 threshold: float = 0.5,
                 graph_threshold: float = 0.8,
                 graph_secondary: float | None = None,
                 graph_secondary_weight: float = 0.5):
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary
        self.graph_secondary_weight = graph_secondary_weight

        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self._build_quantum_filter()

    # ------------------------------------------------------------------
    # Quantum filter construction
    # ------------------------------------------------------------------
    def _build_quantum_filter(self) -> qiskit.QuantumCircuit:
        """Create a parameterised 2‑D quantum filter circuit."""
        n_qubits = self.kernel_size ** 2
        circ = qiskit.QuantumCircuit(n_qubits)
        thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i, theta in enumerate(thetas):
            circ.rx(theta, i)
        circ.barrier()
        circ += random_circuit(n_qubits, 2)
        circ.measure_all()
        circ.thetas = thetas
        return circ

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def run(self, data: np.ndarray) -> float:
        """Execute the quantum filter on a single patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = data.reshape(1, self.kernel_size ** 2)
        param_binds = []
        for dat in data:
            bind = {theta: np.pi if val > self.threshold else 0
                    for theta, val in zip(self.circuit.thetas, dat)}
            param_binds.append(bind)

        job = qiskit.execute(self.circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)

        ones = sum(int(key[i]) for key in counts for i in range(len(key))) * counts[key]
        return ones / (self.shots * self.kernel_size ** 2)

    # ------------------------------------------------------------------
    # State‑vector helpers
    # ------------------------------------------------------------------
    def _statevector(self, data: np.ndarray) -> Statevector:
        """Return the state‑vector after binding the circuit to ``data``."""
        circ = self.circuit.copy()
        bind = {theta: np.pi if val > self.threshold else 0
                for theta, val in zip(circ.thetas, data.reshape(-1))}
        circ.bind_parameters(bind)
        return Statevector.from_instruction(circ)

    @staticmethod
    def _state_fidelity(sv1: Statevector, sv2: Statevector) -> float:
        """Absolute squared overlap between two pure states."""
        return abs(np.vdot(sv1.data, sv2.data)) ** 2

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def fidelity_adjacency(self,
                           states: list[Statevector],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Create a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(si, sj)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def graph_run(self,
                  data_list: list[np.ndarray]) -> nx.Graph:
        """Run a list of patches through the filter and build a fidelity graph."""
        states = [self._statevector(d) for d in data_list]
        return self.fidelity_adjacency(states,
                                       self.graph_threshold,
                                       secondary=self.graph_secondary,
                                       secondary_weight=self.graph_secondary_weight)
