"""
Hybrid convolution module – quantum implementation.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
#  Quantum convolution filter
# --------------------------------------------------------------------------- #
class HybridConvModel(qiskit.providers.Backend):
    """
    Quantum version of the hybrid convolution.  The filter is a
    parametrized circuit that encodes a 2‑D kernel into a set of
    qubits.  The output is the average probability of measuring |1>
    across all qubits after applying the circuit to the data‑encoded
    state.  The class inherits from `qiskit.providers.Backend` only
    to expose a `run` method with a familiar signature.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 200,
        threshold: float = 0.5,
        backend_name: str = "qasm_simulator",
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = qiskit.Aer.get_backend(backend_name)
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits)
        thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, theta in enumerate(thetas):
            qc.rx(theta, i)
        qc.barrier()
        qc += random_circuit(self.n_qubits, 2)
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on classical data.  Each element of the flattened
        kernel is mapped to a rotation angle; values above `threshold`
        trigger a π rotation.  The measurement statistics are then
        aggregated into a single scalar.
        """
        param_binds = []
        for i, val in enumerate(data.flatten()):
            bind = {self.circuit.parameters[i]: np.pi if val > self.threshold else 0}
            param_binds.append(bind)

        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute average number of |1> per qubit
        total_ones = 0
        for bitstring, cnt in result.items():
            ones = sum(int(bit) for bit in bitstring)
            total_ones += ones * cnt
        return total_ones / (self.shots * self.n_qubits)

    # --------------------------------------------------------------------- #
    #  Graph utilities – fidelity between quantum states
    # --------------------------------------------------------------------- #
    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Absolute squared overlap between two pure states."""
        return np.abs(a.data.conj().dot(b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: list[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from pairwise fidelities of quantum states.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s1), (j, s2) in itertools.combinations(enumerate(states), 2):
            fid = HybridConvModel.state_fidelity(s1, s2)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = [
    "HybridConvModel",
]
